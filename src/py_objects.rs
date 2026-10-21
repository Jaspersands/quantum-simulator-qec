//! The compiled objects behind the public Python API (`python/stabilizer_qec`). Each parses or
//! builds once and is reused; shots go in and results come out as Stim's b8 rows and
//! little-endian arrays, which the Python layer turns into numpy arrays; every loop over shots
//! releases the GIL. The shot loops here also serve the 0.4 functions in `py_api`.

// PyO3's #[pymethods] expansion converts each PyResult's error into PyErr, which clippy flags
// as a useless conversion; it is the macro's, not this code's.
#![allow(clippy::useless_conversion)]

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

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
    /// Stim's text, neighbouring instructions fused as Stim's parser fuses them; with `fuse`
    /// false each line stays its own instruction (a circuit written from its items).
    #[new]
    #[pyo3(signature = (text, fuse=true))]
    fn new(text: &str, fuse: bool) -> PyResult<Self> {
        let circuit = if fuse { crate::api::Circuit::parse(text) } else { crate::api::Circuit::parse_unfused(text) };
        Ok(PyCircuit { circuit: circuit.map_err(api_err)? })
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
    #[pyo3(signature = (decompose, approximate=None, flatten=false, ignore_failures=false, allow_gauge=false, block_remnants=false, deltakit=false))]
    #[allow(clippy::too_many_arguments)]
    fn detector_error_model(&self, decompose: bool, approximate: Option<f64>, flatten: bool, ignore_failures: bool, allow_gauge: bool, block_remnants: bool, deltakit: bool) -> PyResult<PyDem> {
        let options = crate::api::DemOptions::new()
            .deltakit_leakage(deltakit)
            .decompose_errors(decompose)
            .approximate_disjoint_errors(approximate)
            .flatten_loops(flatten)
            .ignore_decomposition_failures(ignore_failures)
            .allow_gauge_detectors(allow_gauge)
            .block_decomposition_from_introducing_remnant_edges(block_remnants);
        Ok(PyDem { dem: self.circuit.detector_error_model(&options).map_err(api_err)? })
    }

    /// The noiseless reference run's measurement record (the one the converter compares with),
    /// as one b8 row.
    fn reference_sample<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let inner = &self.circuit.inner;
        let qubits = crate::batch_sampler::Counts::of(&inner.instrs).map_err(err)?.qubits;
        let bits = py.detach(|| crate::m2d::reference(inner, qubits));
        let mut row = vec![0u8; bits.len().div_ceil(8)];
        for (k, &b) in bits.iter().enumerate() {
            if b {
                row[k / 8] |= 1 << (k % 8);
            }
        }
        Ok(PyBytes::new(py, &row))
    }

    /// A sampler of raw measurement records (`skip_reference`: flips from all zeros), on a
    /// `device`: "cpu" (64 shots a word), "cpu-lanes" (32 shots a word, integer draws: the
    /// GPU's reference) or "gpu".
    #[pyo3(signature = (seed, skip_reference, device="cpu"))]
    fn measurement_sampler(&self, py: Python<'_>, seed: u64, skip_reference: bool, device: &str) -> PyResult<PyMeasurementSampler> {
        let inner = &self.circuit.inner;
        let sampler = BatchSampler::new(inner).map_err(err)?;
        let lanes = lanes_for(device, &sampler)?;
        let reference = if skip_reference {
            Vec::new()
        } else {
            let qubits = crate::batch_sampler::Counts::of(&inner.instrs).map_err(err)?.qubits;
            py.detach(|| crate::m2d::reference(inner, qubits))
        };
        Ok(PyMeasurementSampler { sampler, lanes, device: device.to_string(), reference, seed, next: 0.into() })
    }

    #[pyo3(signature = (seed, device="cpu"))]
    fn sampler(&self, seed: u64, device: &str) -> PyResult<PySampler> {
        let sampler = BatchSampler::new(&self.circuit.inner).map_err(err)?;
        let lanes = lanes_for(device, &sampler)?;
        Ok(PySampler { sampler, lanes, device: device.to_string(), seed, next: 0.into() })
    }

    fn m2d(&self) -> PyResult<PyM2d> {
        Ok(PyM2d { m2d: M2d::new(&self.circuit.inner).map_err(err)? })
    }

    /// The per-shot frame sampler, which runs leakage (Deltakit's Stim exactly, `deltakit`).
    #[pyo3(signature = (seed, leaked_reads_one=true, deltakit=false))]
    fn leakage_sampler(&self, seed: u64, leaked_reads_one: bool, deltakit: bool) -> PyResult<PyLeakageSampler> {
        if deltakit {
            let sampler = crate::frame_sampler::FrameSampler::new_deltakit(&self.circuit.inner).map_err(err)?;
            return Ok(PyLeakageSampler { sampler, seed, next: 0.into() });
        }
        let mut sampler = crate::frame_sampler::FrameSampler::new(&self.circuit.inner).and_then(|s| s.with_reference(&self.circuit.inner)).map_err(err)?;
        sampler.leaked_reads_one = leaked_reads_one;
        Ok(PyLeakageSampler { sampler, seed, next: 0.into() })
    }

    /// The exact state-vector sampler.
    fn exact_sampler(&self, seed: u64) -> PyResult<PyExactSampler> {
        Ok(PyExactSampler { program: crate::statevec::Program::new(&self.circuit.inner).map_err(err)?, seed, next: 0.into() })
    }

    /// The stabilizer-rank sampler: Clifford+T circuits at any qubit count.
    fn stabilizer_rank_sampler(&self, py: Python<'_>, seed: u64) -> PyResult<PyStabilizerRankSampler> {
        let inner = &self.circuit.inner;
        let program = py.detach(|| crate::stabrank::Program::new(inner)).map_err(err)?;
        Ok(PyStabilizerRankSampler { program, seed, next: 0.into(), peak: 0.into() })
    }

    /// `exact_distribution` by the stabilizer-rank engine.
    fn stabilizer_rank_distribution(&self, py: Python<'_>, max_branches: usize) -> PyResult<Vec<(Vec<bool>, u64, f64)>> {
        let program = crate::stabrank::Program::new(&self.circuit.inner).map_err(err)?;
        let d = py.detach(|| program.distribution(max_branches)).map_err(err)?;
        Ok(d.into_iter().map(|((dets, obs), p)| (dets, obs, p)).collect())
    }

    /// How many non-Clifford gates the circuit has (T, and rotations written as gates at angles
    /// that are not Clifford); coherent errors are not counted.
    fn num_non_clifford_gates(&self) -> PyResult<usize> {
        crate::stabrank::non_clifford_gates(&self.circuit.inner).map_err(err)
    }

    /// How many coherent errors are at Clifford angles (fixed Cliffords the joint sampler runs exactly).
    fn num_clifford_angle_errors(&self) -> usize {
        crate::stabrank::clifford_angle_errors(&self.circuit.inner)
    }

    /// Tsim's T-count.
    fn t_count(&self) -> PyResult<usize> {
        crate::stabrank::t_count(&self.circuit.inner).map_err(err)
    }

    /// Whether every gate and coherent error is Clifford.
    fn is_clifford(&self) -> PyResult<bool> {
        crate::stabrank::is_clifford(&self.circuit.inner).map_err(err)
    }

    /// The circuit's unitary: (n, 2^n × 2^n complex128 entries, row-major).
    fn unitary_matrix<'py>(&self, py: Python<'py>, little_endian: bool) -> PyResult<(usize, Bound<'py, PyBytes>)> {
        let inner = &self.circuit.inner;
        let (n, m) = py.detach(|| crate::statevec::unitary_matrix(inner, little_endian)).map_err(err)?;
        Ok((n, PyBytes::new(py, &le_f64(m.iter().flat_map(|c| [c.re, c.im])))))
    }

    /// Tsim's matrix: (n, k, 2^n × 2^k complex128 entries, row-major), unused qubits dropped
    /// and a qubit reset first an output alone.
    fn tsim_matrix<'py>(&self, py: Python<'py>, little_endian: bool) -> PyResult<(usize, usize, Bound<'py, PyBytes>)> {
        let inner = &self.circuit.inner;
        let (n, k, m) = py.detach(|| crate::statevec::tsim_matrix(inner, little_endian)).map_err(err)?;
        Ok((n, k, PyBytes::new(py, &le_f64(m.iter().flat_map(|c| [c.re, c.im])))))
    }

    /// The coherent sampler: the twirl, reweighted by the interference it leaves out.
    #[pyo3(signature = (seed, order=3, max_cluster=12, quads=true, hops=1, max_complete=crate::coherent::kernel::MAX_COMPLETE, max_contraction=crate::coherent::kernel::MAX_CONTRACTION))]
    #[allow(clippy::too_many_arguments)]
    fn coherent_sampler(&self, py: Python<'_>, seed: u64, order: usize, max_cluster: usize, quads: bool, hops: usize, max_complete: usize, max_contraction: usize) -> PyResult<PyCoherentSampler> {
        let options = crate::coherent::kernel::CoherentOptions { order, max_cluster, quads, hops, max_complete, max_contraction };
        let inner = &self.circuit.inner;
        let sampler = py.detach(|| crate::coherent::sampler::CoherentSampler::new(inner, options)).map_err(err)?;
        Ok(PyCoherentSampler { sampler, seed, next: 0.into() })
    }

    /// The joint sampler: non-Clifford gates exact, coherent errors twirled and reweighted.
    #[pyo3(signature = (seed, order=3, max_cluster=12, quads=true, hops=1, pairs=false, max_complete=crate::coherent::kernel::MAX_COMPLETE))]
    fn joint_sampler(&self, py: Python<'_>, seed: u64, order: usize, max_cluster: usize, quads: bool, hops: usize, pairs: bool, max_complete: usize) -> PyResult<PyJointSampler> {
        let options = crate::coherent::kernel::CoherentOptions { order, max_cluster, quads, hops, max_complete, ..Default::default() };
        let inner = &self.circuit.inner;
        let sampler = py.detach(|| crate::coherent::joint::JointSampler::new(inner, options).map(|s| s.with_pairs(pairs))).map_err(err)?;
        Ok(PyJointSampler { sampler, seed, next: 0.into() })
    }

    /// The joint sampler's exact distribution with the whole kernel, for small circuits:
    /// (detection events, observable flips, probability).
    fn joint_distribution(&self, py: Python<'_>, max_branches: usize) -> PyResult<Vec<(Vec<bool>, u64, f64)>> {
        let inner = &self.circuit.inner;
        let d = py.detach(|| crate::coherent::joint::JointSampler::with_whole_kernel(inner).and_then(|s| s.exact_distribution(max_branches))).map_err(err)?;
        Ok(d.into_iter().map(|((dets, obs), p)| (dets, obs, p)).collect())
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

    /// The final density matrix over qubits 0..n (n, its entries as complex128 bytes),
    /// records averaged away.
    fn density_matrix<'py>(&self, py: Python<'py>, max_branches: usize) -> PyResult<(usize, Bound<'py, PyBytes>)> {
        let program = crate::density::DensityProgram::with_layout(&self.circuit.inner, true, crate::statevec::Outcomes::StateOnly).map_err(err)?;
        let rho = py.detach(|| program.final_state(None, max_branches)).map_err(err)?;
        Ok((rho.num_qubits(), density_bytes(py, &rho)))
    }

    /// `exact_distribution` by the density matrix: noise channels applied whole, branching only
    /// on records.
    fn density_distribution(&self, py: Python<'_>, max_branches: usize) -> PyResult<Vec<(Vec<bool>, u64, f64)>> {
        let program = crate::density::DensityProgram::new(&self.circuit.inner).map_err(err)?;
        let d = py.detach(|| program.distribution(max_branches)).map_err(err)?;
        Ok(d.into_iter().map(|((dets, obs), p)| (dets, obs, p)).collect())
    }

    /// The perturbative expansion to `order` faults: (the number of detectors, each outcome as
    /// its flipped detectors then the number of detectors plus each flipped observable, with its
    /// weight, one number or, with `degree`, a polynomial's coefficients up to x^degree).
    fn perturbative_distribution(&self, py: Python<'_>, order: usize, degree: Option<usize>, max_terms: usize) -> PyResult<(usize, Vec<(Vec<u32>, Vec<f64>)>)> {
        let sites = crate::perturb::sites(&self.circuit.inner).map_err(err)?;
        let out = py.detach(|| crate::perturb::expand(&sites.sites, order, degree, max_terms)).map_err(err)?;
        Ok((sites.num_detectors, out))
    }

    /// The fault sites the perturbative expansion sums over: per site, each fault's
    /// probability and effect (as `perturbative_distribution`'s outcomes).
    fn perturbative_sites(&self) -> PyResult<Vec<Vec<(f64, Vec<u32>)>>> {
        let sites = crate::perturb::sites(&self.circuit.inner).map_err(err)?;
        Ok(sites.sites.into_iter().map(|s| s.faults).collect())
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

    #[getter]
    fn num_measurements(&self) -> usize {
        self.sampler.num_measurements()
    }

    /// Measurement records b8, one row per shot (the stream `sample` draws from).
    fn sample_measurements<'py>(&self, py: Python<'py>, shots: usize, threads: usize) -> Bound<'py, PyBytes> {
        let first = self.next.fetch_add(shots as u64, std::sync::atomic::Ordering::Relaxed);
        let (sampler, seed) = (&self.sampler, self.seed);
        let w = sampler.num_measurements().div_ceil(8);
        let rows: Vec<Vec<u8>> = py.detach(|| {
            crate::parallel::parallel(shots, threads, |range| {
                range
                    .map(|s| {
                        let mut rng = crate::surface_code::Xorshift::new(crate::batch_sampler::batch_seed(seed, first + s as u64));
                        let mut row = vec![0u8; w];
                        for (k, b) in sampler.sample_measurements(&mut rng).0.into_iter().enumerate() {
                            row[k / 8] |= (b as u8) << (k % 8);
                        }
                        row
                    })
                    .collect()
            })
        });
        PyBytes::new(py, &rows.concat())
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

    #[getter]
    fn num_nonlocal_generators(&self) -> usize {
        self.sampler.kernel.num_nonlocal
    }

    #[getter]
    fn kernel_complete(&self) -> bool {
        self.sampler.kernel.complete
    }

    #[getter]
    fn exact_weights(&self) -> bool {
        self.sampler.kernel.is_whole()
    }

    #[getter]
    fn contraction_width(&self) -> Option<usize> {
        self.sampler.kernel.contraction_width()
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

#[pyclass(name = "JointSampler", module = "stabilizer_qec._core")]
pub struct PyJointSampler {
    sampler: crate::coherent::joint::JointSampler,
    seed: u64,
    next: std::sync::atomic::AtomicU64,
}

#[pymethods]
impl PyJointSampler {
    #[getter]
    fn num_detectors(&self) -> usize {
        self.sampler.num_detectors()
    }

    #[getter]
    fn num_observables(&self) -> usize {
        self.sampler.num_observables()
    }

    #[getter]
    fn num_locations(&self) -> usize {
        self.sampler.num_locations()
    }

    #[getter]
    fn num_exact(&self) -> usize {
        self.sampler.num_exact()
    }

    #[getter]
    fn num_generators(&self) -> usize {
        self.sampler.num_generators()
    }

    #[getter]
    fn num_nonlocal_generators(&self) -> usize {
        self.sampler.num_nonlocal_generators()
    }

    #[getter]
    fn kernel_complete(&self) -> bool {
        self.sampler.kernel_complete()
    }

    #[getter]
    fn exact_weights(&self) -> bool {
        self.sampler.is_whole()
    }

    fn sample<'py>(&self, py: Python<'py>, shots: usize, threads: usize) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>, Vec<f64>) {
        let first = self.next.fetch_add(shots as u64, std::sync::atomic::Ordering::Relaxed);
        let (sampler, seed) = (&self.sampler, self.seed);
        let b = py.detach(|| sampler.sample_seeded(seed, first, shots, threads));
        (PyBytes::new(py, &b.detectors), PyBytes::new(py, &b.observables), b.weights)
    }
}

#[pyclass(name = "StabilizerRankSampler", module = "stabilizer_qec._core")]
pub struct PyStabilizerRankSampler {
    program: crate::stabrank::Program,
    seed: u64,
    next: std::sync::atomic::AtomicU64,
    peak: std::sync::atomic::AtomicUsize,
}

#[pymethods]
impl PyStabilizerRankSampler {
    #[getter]
    fn num_detectors(&self) -> usize {
        self.program.num_detectors()
    }

    #[getter]
    fn num_observables(&self) -> usize {
        self.program.num_observables()
    }

    #[getter]
    fn num_measurements(&self) -> usize {
        self.program.num_measurements()
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.program.num_qubits()
    }

    #[getter]
    fn num_non_clifford(&self) -> usize {
        self.program.num_non_clifford()
    }

    #[getter]
    fn peak_terms(&self) -> usize {
        self.peak.load(std::sync::atomic::Ordering::Relaxed)
    }

    #[getter]
    fn is_noisy(&self) -> bool {
        self.program.is_noisy()
    }

    fn sample<'py>(&self, py: Python<'py>, shots: usize, threads: usize) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>) {
        let first = self.next.fetch_add(shots as u64, std::sync::atomic::Ordering::Relaxed);
        let (program, seed) = (&self.program, self.seed);
        let (d, o, peak) = py.detach(|| crate::stabrank::sample_seeded(program, seed, first, shots, threads));
        self.peak.fetch_max(peak, std::sync::atomic::Ordering::Relaxed);
        (PyBytes::new(py, &d), PyBytes::new(py, &o))
    }

    /// `shots` measurement records as b8 rows, from the same stream of shots.
    fn sample_measurements<'py>(&self, py: Python<'py>, shots: usize, threads: usize) -> Bound<'py, PyBytes> {
        let first = self.next.fetch_add(shots as u64, std::sync::atomic::Ordering::Relaxed);
        let (program, seed) = (&self.program, self.seed);
        let (m, peak) = py.detach(|| crate::stabrank::measurements_seeded(program, seed, first, shots, threads));
        self.peak.fetch_max(peak, std::sync::atomic::Ordering::Relaxed);
        PyBytes::new(py, &m)
    }

    /// The probability of each record (rows of bools), exact without noise, else averaged over
    /// `samples` draws of the noise from the stream `seed`.
    fn probability_of(&self, py: Python<'_>, records: Vec<Vec<bool>>, samples: usize, seed: u64) -> PyResult<Vec<f64>> {
        let program = &self.program;
        py.detach(|| records.iter().map(|r| program.probability_of(r, samples, seed)).collect::<Result<Vec<f64>, String>>()).map_err(err)
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

    /// The model's text with every number exact, for copies (Stim's text rounds them).
    fn exact_text(&self) -> String {
        self.dem.exact_text()
    }

    /// Appends the instructions of a piece of model text.
    fn append_text(&mut self, text: &str) -> PyResult<()> {
        self.dem.append_text(text).map_err(api_err)
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
    /// The lane program, on "cpu-lanes" and "gpu".
    lanes: Option<crate::lanes::LaneProgram>,
    device: String,
    seed: u64,
    /// The next batch's index: the shots drawn so far, in batches of 64 (32 on the lane
    /// devices). Each call reserves its batches before it samples, so threads sharing the
    /// sampler draw disjoint batches.
    next: std::sync::atomic::AtomicU64,
}

/// The lane program a device needs: none for "cpu", the program for "cpu-lanes" and "gpu" (the
/// latter only when the `gpu` feature is built and a device answers).
fn lanes_for(device: &str, sampler: &BatchSampler) -> PyResult<Option<crate::lanes::LaneProgram>> {
    match device {
        "cpu" => Ok(None),
        "cpu-lanes" => Ok(Some(crate::lanes::LaneProgram::from_sampler(sampler).map_err(err)?)),
        "gpu" => {
            #[cfg(feature = "gpu")]
            {
                crate::gpu::device_ready().map_err(err)?;
                Ok(Some(crate::lanes::LaneProgram::from_sampler(sampler).map_err(err)?))
            }
            #[cfg(not(feature = "gpu"))]
            {
                Err(err("device 'gpu' needs the package built with its gpu feature (pip install stabilizer-qec[gpu])".to_string()))
            }
        }
        other => Err(err(format!("device must be 'cpu', 'cpu-lanes' or 'gpu', not {other:?}"))),
    }
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

    #[getter]
    fn device(&self) -> &str {
        &self.device
    }

    /// `shots` shots as b8 rows of detectors and of observables. A call that ends mid-batch
    /// discards the batch's other lanes; the next call starts a new batch.
    fn sample<'py>(&self, py: Python<'py>, shots: usize, threads: usize) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let per_batch = if self.lanes.is_some() { crate::lanes::LANES } else { 64 };
        let first = self.next.fetch_add(shots.div_ceil(per_batch) as u64, std::sync::atomic::Ordering::Relaxed);
        let (sampler, seed) = (&self.sampler, self.seed);
        let (d, o) = match (&self.lanes, self.device.as_str()) {
            (None, _) => py.detach(|| sampler.sample_seeded(seed, first, shots, threads)),
            #[cfg(feature = "gpu")]
            (Some(lanes), "gpu") => py.detach(|| crate::gpu::sample_seeded(lanes, seed, first, shots)).map_err(err)?,
            (Some(lanes), _) => py.detach(|| lanes.sample_seeded(seed, first, shots, threads)),
        };
        Ok((PyBytes::new(py, &d), PyBytes::new(py, &o)))
    }
}

/// Raw measurement records, batch after batch: a noiseless reference run with each shot's flips.
#[pyclass(name = "MeasurementSampler", module = "stabilizer_qec._core")]
pub struct PyMeasurementSampler {
    sampler: BatchSampler,
    lanes: Option<crate::lanes::LaneProgram>,
    device: String,
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

    #[getter]
    fn device(&self) -> &str {
        &self.device
    }

    /// `shots` measurement records as b8 rows.
    fn sample<'py>(&self, py: Python<'py>, shots: usize, threads: usize) -> PyResult<Bound<'py, PyBytes>> {
        let per_batch = if self.lanes.is_some() { crate::lanes::LANES } else { 64 };
        let first = self.next.fetch_add(shots.div_ceil(per_batch) as u64, std::sync::atomic::Ordering::Relaxed);
        let (sampler, seed, reference) = (&self.sampler, self.seed, &self.reference);
        let rows = match (&self.lanes, self.device.as_str()) {
            (None, _) => py.detach(|| sampler.sample_measurements_seeded(seed, first, shots, threads, reference)),
            #[cfg(feature = "gpu")]
            (Some(lanes), "gpu") => py.detach(|| crate::gpu::sample_measurements_seeded(lanes, seed, first, shots, reference)).map_err(err)?,
            (Some(lanes), _) => py.detach(|| lanes.sample_measurements_seeded(seed, first, shots, threads, reference)),
        };
        Ok(PyBytes::new(py, &rows))
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

    /// The edges one shot's matching uses (correlated: pass two's), as node pairs with -1 for
    /// the boundary.
    fn decode_to_edges(&self, defects: Vec<u32>) -> PyResult<Vec<(i64, i64)>> {
        let mut scratch = crate::sparse::Scratch::new(&self.graph);
        let mut defects = defects;
        defects.sort_unstable();
        let ids = match &self.corr {
            Some(c) => self.graph.decode_correlated_edge_ids(c, &mut scratch, &defects),
            None => self.graph.decode_edge_ids(&mut scratch, &defects),
        }
        .map_err(|_| PyValueError::new_err("No perfect matching could be found. This likely means that the syndrome has odd parity in the support of a connected component without a boundary."))?;
        let n = self.graph.num_nodes as u32;
        let end = |v: u32| if v == n { -1 } else { v as i64 };
        Ok(ids.iter().map(|&id| {
            let (u, v) = self.graph.edge_ends(id);
            (end(u), end(v))
        }).collect())
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
    #[pyo3(signature = (dem, max_iter, method, scale, osd, order, sched="parallel"))]
    fn new(dem: &PyDem, max_iter: usize, method: &str, scale: f64, osd: &str, order: usize, sched: &str) -> PyResult<Self> {
        let d = dem.flat()?;
        let columns: Vec<Vec<u32>> = d.mechanisms.iter().map(|m| m.detectors.clone()).collect();
        let priors: Vec<f64> = d.mechanisms.iter().map(|m| m.p).collect();
        let mut dec = BpOsd::new(d.num_detectors, columns, &priors, bp_method(method, scale)?, max_iter, osd_method(osd, order)?)
            .map_err(err)?;
        dec.schedule = schedule(sched)?;
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
    #[pyo3(signature = (dem, max_iter, method, scale, lsd, order, bits_per_step, sched="parallel"))]
    fn new(dem: &PyDem, max_iter: usize, method: &str, scale: f64, lsd: &str, order: usize, bits_per_step: usize, sched: &str) -> PyResult<Self> {
        let d = dem.flat()?;
        let columns: Vec<Vec<u32>> = d.mechanisms.iter().map(|m| m.detectors.clone()).collect();
        let priors: Vec<f64> = d.mechanisms.iter().map(|m| m.p).collect();
        // As ldpc: no iteration limit given means one per fault.
        let max_iter = if max_iter == 0 { columns.len() } else { max_iter };
        let mut dec = BpLsd::new(d.num_detectors, columns, &priors, bp_method(method, scale)?, max_iter, osd_method(lsd, order)?, bits_per_step, false).map_err(err)?;
        dec.schedule = schedule(sched)?;
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

/// Ambiguity Clustering on an undecomposed model: each fault a column, its prior the model's.
#[pyclass(name = "DemAmbiguityClustering", module = "stabilizer_qec._core")]
pub struct PyDemAmbiguityClustering {
    dec: crate::ambiguity::AmbiguityClustering,
    #[pyo3(get)]
    num_detectors: usize,
    #[pyo3(get)]
    num_observables: usize,
}

#[pymethods]
impl PyDemAmbiguityClustering {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(dem: &PyDem, max_iter: usize, method: &str, scale: f64, kappa: f64, schedule: &str) -> PyResult<Self> {
        let d = dem.flat()?;
        let columns: Vec<Vec<u32>> = d.mechanisms.iter().map(|m| m.detectors.clone()).collect();
        let priors: Vec<f64> = d.mechanisms.iter().map(|m| m.p).collect();
        let obs = d.mechanisms.iter().map(|m| m.observables).collect();
        let extra = (kappa * columns.len() as f64).round() as usize;
        let mut dec = crate::ambiguity::AmbiguityClustering::new(d.num_detectors, columns, obs, &priors, bp_method(method, scale)?, max_iter, extra).map_err(err)?;
        dec.schedule = match schedule {
            "parallel" => crate::bp::Schedule::Parallel,
            "serial" => crate::bp::Schedule::Serial,
            "serial_relative" => crate::bp::Schedule::SerialRelative,
            other => return Err(err(format!("schedule '{other}' is not parallel, serial or serial_relative"))),
        };
        Ok(PyDemAmbiguityClustering { dec, num_detectors: d.num_detectors, num_observables: d.num_observables })
    }

    /// K, the columns stage 2 adds.
    #[getter]
    fn extra_columns(&self) -> usize {
        self.dec.extra_columns
    }

    /// (observables as u64 per shot, one byte per shot: 1 where BP converged, 0 where AC ran,
    /// 2 where no correction explains the shot; per shot the blocks, the ambiguous blocks and
    /// the solutions searched, as u32s).
    fn decode_batch<'py>(&self, py: Python<'py>, packed: &[u8], shots: usize, threads: usize) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let nd = self.num_detectors;
        check_rows(packed.len(), nd.div_ceil(8), shots, nd, "detectors")?;
        let dec = &self.dec;
        let out = py.detach(|| crate::batch::ambiguity_shots(dec, packed, nd, shots, threads));
        let flags: Vec<u8> = out.iter().map(|(_, o)| if o.failed { 2 } else { u8::from(o.converged) }).collect();
        let stats: Vec<u8> = out.iter().flat_map(|(_, o)| [o.blocks, o.ambiguous_blocks, o.solutions].into_iter().flat_map(|x| (x as u32).to_le_bytes())).collect();
        Ok((PyBytes::new(py, &le_u64(out.iter().map(|x| x.0))), PyBytes::new(py, &flags), PyBytes::new(py, &stats)))
    }
}

/// The weights a clustering decoder grows by: every edge 2 steps (`None`), or each edge's
/// weight in at most `resolution` steps.
fn cluster_graph(dem: &PyDem, resolution: Option<u32>) -> PyResult<(crate::clustering::ClusterGraph, usize, usize)> {
    let d = dem.flat()?;
    if resolution == Some(0) {
        return Err(err("weight_resolution must be at least 1".to_string()));
    }
    let (graph, _) = DemDecoder::new(d).map_err(err)?.into_parts(false);
    Ok((crate::clustering::ClusterGraph::new(&graph, resolution), d.num_detectors, d.num_observables))
}

fn cluster_stats(stats: impl Iterator<Item = crate::clustering::ClusterStats>) -> Vec<u8> {
    stats.flat_map(|s| [s.rounds, s.steps.min(u64::from(u32::MAX)) as u32].into_iter().flat_map(u32::to_le_bytes)).collect()
}

/// Collision Clustering (Barber et al. 2023) on a decomposed model's matching graph.
#[pyclass(name = "DemCollisionClustering", module = "stabilizer_qec._core")]
pub struct PyDemCollisionClustering {
    dec: crate::clustering::CollisionClustering,
    #[pyo3(get)]
    num_detectors: usize,
    #[pyo3(get)]
    num_observables: usize,
}

#[pymethods]
impl PyDemCollisionClustering {
    #[new]
    #[pyo3(signature = (dem, resolution=None))]
    fn new(dem: &PyDem, resolution: Option<u32>) -> PyResult<Self> {
        let (graph, num_detectors, num_observables) = cluster_graph(dem, resolution)?;
        Ok(PyDemCollisionClustering { dec: crate::clustering::CollisionClustering::new(graph), num_detectors, num_observables })
    }

    /// (observables as u64 per shot, the shots with no correction, per shot the rounds of
    /// growth and the pairs compared as u32s).
    fn decode_batch<'py>(&self, py: Python<'py>, packed: &[u8], shots: usize, threads: usize) -> PyResult<(Bound<'py, PyBytes>, Vec<usize>, Bound<'py, PyBytes>)> {
        let nd = self.num_detectors;
        check_rows(packed.len(), nd.div_ceil(8), shots, nd, "detectors")?;
        let dec = &self.dec;
        let out = py.detach(|| crate::batch::collision_shots(dec, packed, nd, shots, threads));
        let failed = out.iter().enumerate().filter(|(_, o)| o.0.is_none()).map(|(s, _)| s).collect();
        Ok((PyBytes::new(py, &le_u64(out.iter().map(|o| o.0.unwrap_or(0)))), failed, PyBytes::new(py, &cluster_stats(out.iter().map(|o| o.1)))))
    }
}

/// The Local Clustering Decoder (Ziad et al. 2024) on a decomposed model's matching graph,
/// with its adaptivity map.
#[pyclass(name = "DemLocalClustering", module = "stabilizer_qec._core")]
pub struct PyDemLocalClustering {
    dec: crate::clustering::LocalClustering,
    #[pyo3(get)]
    num_detectors: usize,
    #[pyo3(get)]
    num_observables: usize,
}

#[pymethods]
impl PyDemLocalClustering {
    /// `adaptivity`: per trigger, the edges it pre-grows as detector pairs (`num_detectors`
    /// for the boundary); pairs with no edge in the graph are left out.
    #[new]
    #[pyo3(signature = (dem, resolution=None, adaptivity=Vec::new()))]
    fn new(dem: &PyDem, resolution: Option<u32>, adaptivity: Vec<Vec<(u32, u32)>>) -> PyResult<Self> {
        let (graph, num_detectors, num_observables) = cluster_graph(dem, resolution)?;
        let map = adaptivity.iter().map(|edges| {
            let mut ids: Vec<u32> = edges.iter().filter_map(|&(u, v)| graph.edge_id(u, v)).collect();
            ids.sort_unstable();
            ids.dedup();
            ids
        }).collect();
        Ok(PyDemLocalClustering { dec: crate::clustering::LocalClustering::new(graph, map), num_detectors, num_observables })
    }

    /// The graph's edges as detector pairs (`num_detectors` for the boundary).
    #[getter]
    fn edges(&self) -> Vec<(u32, u32)> {
        self.dec.graph.ends.clone()
    }

    /// Per trigger, how many edges it pre-grows.
    #[getter]
    fn adaptivity_sizes(&self) -> Vec<usize> {
        self.dec.adaptivity.iter().map(Vec::len).collect()
    }

    /// (observables as u64 per shot, the shots with no correction, per shot the rounds of
    /// growth, the stage passes and the edges pre-grown as u32s). `heralds`: b8 rows of
    /// `num_heralds` bits per shot, firing those triggers.
    #[pyo3(signature = (packed, shots, threads, heralds=None, num_heralds=0))]
    fn decode_batch<'py>(&self, py: Python<'py>, packed: &[u8], shots: usize, threads: usize, heralds: Option<&[u8]>, num_heralds: usize) -> PyResult<(Bound<'py, PyBytes>, Vec<usize>, Bound<'py, PyBytes>)> {
        let nd = self.num_detectors;
        check_rows(packed.len(), nd.div_ceil(8), shots, nd, "detectors")?;
        if let Some(h) = heralds {
            check_rows(h.len(), num_heralds.div_ceil(8), shots, num_heralds, "heralds")?;
        }
        let dec = &self.dec;
        let out = py.detach(|| crate::batch::local_clustering_shots(dec, packed, nd, heralds.map(|h| (h, num_heralds)), shots, threads));
        let failed = out.iter().enumerate().filter(|(_, o)| o.0.is_none()).map(|(s, _)| s).collect();
        let stats: Vec<u8> = out.iter().flat_map(|o| [o.1.rounds, o.1.steps.min(u64::from(u32::MAX)) as u32, o.2].into_iter().flat_map(u32::to_le_bytes)).collect();
        Ok((PyBytes::new(py, &le_u64(out.iter().map(|o| o.0.unwrap_or(0)))), failed, PyBytes::new(py, &stats)))
    }
}

/// Argument reweighting (Xie et al. 2026) around matching, union-find, BP+OSD, BP+LSD or
/// Relay-BP.
/// A decoder whose priors a shot can change: matching or union-find on the model's graph, or
/// the BP decoder's core `inner` (a `DemBpOsd`, `DemBpLsd` or `DemRelay`) for those kinds.
fn prior_inner(d: &Dem, kind: &str, inner: Option<&Bound<'_, PyAny>>) -> PyResult<crate::reweight::Inner> {
    use crate::reweight::Inner;
    let need = |what: &str| err(format!("the {what} decoder's core is needed"));
    Ok(match kind {
        "matching" | "union_find" => {
            let (graph, _) = DemDecoder::new(d).map_err(err)?.into_parts(false);
            if kind == "matching" { Inner::Matching(graph) } else { Inner::UnionFind(graph) }
        }
        "bposd" => {
            let x = inner.ok_or_else(|| need("bposd"))?.cast::<PyDemBpOsd>().map_err(|e| err(e.to_string()))?.borrow();
            Inner::BpOsd(x.dec.clone(), x.obs.clone())
        }
        "bplsd" => {
            let x = inner.ok_or_else(|| need("bplsd"))?.cast::<PyDemBpLsd>().map_err(|e| err(e.to_string()))?.borrow();
            Inner::BpLsd(x.dec.clone(), x.obs.clone())
        }
        "relay_bp" => {
            let x = inner.ok_or_else(|| need("relay_bp"))?.cast::<PyDemRelay>().map_err(|e| err(e.to_string()))?.borrow();
            Inner::Relay(x.dec.clone(), x.obs.clone(), x.dec.config.seed)
        }
        other => return Err(err(format!("decoder '{other}' is not matching, union_find, bposd, bplsd or relay_bp"))),
    })
}

/// The AI pre-decoder: the layout of a model's detectors and faults on a grid, the network, its
/// optimizer's state, and the model's sampler for training.
/// PECOS's sparse stabilizer simulator (`stabilizer_qec::pecos_sparse`), which
/// `stabilizer_qec.pecos`'s `SparseSim` wraps as PECOS's Cython wraps its C++.
#[pyclass(name = "PecosSparseSim", module = "stabilizer_qec._core")]
pub struct PyPecosSparseSim {
    sim: crate::pecos_sparse::SparseSim,
}

fn pecos_sets(py: Python<'_>, sets: &[crate::pecos_sparse::Set]) -> PyResult<Py<PyAny>> {
    let list = pyo3::types::PyList::empty(py);
    for s in sets {
        let mut v: Vec<u32> = s.iter().copied().collect();
        v.sort_unstable();
        list.append(pyo3::types::PySet::new(py, &v)?)?;
    }
    Ok(list.into_any().unbind())
}

fn pecos_generators(py: Python<'_>, g: &crate::pecos_sparse::Generators) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item("col_x", pecos_sets(py, &g.col_x)?)?;
    d.set_item("col_z", pecos_sets(py, &g.col_z)?)?;
    d.set_item("row_x", pecos_sets(py, &g.row_x)?)?;
    d.set_item("row_z", pecos_sets(py, &g.row_z)?)?;
    Ok(d.into_any().unbind())
}

#[pymethods]
impl PyPecosSparseSim {
    #[new]
    fn new(num_qubits: usize) -> Self {
        PyPecosSparseSim { sim: crate::pecos_sparse::SparseSim::new(num_qubits) }
    }
    #[getter]
    fn num_qubits(&self) -> usize {
        self.sim.num_qubits
    }
    fn clear(&mut self) {
        self.sim.clear();
    }
    /// The gate taking X and Z on `qubits[j]` to `images[2j]` and `images[2j + 1]`, each
    /// `(phase, x bits, z bits)` for `i^phase prod X^x Z^z` over the gate's qubits.
    fn gate(&mut self, qubits: Vec<usize>, images: Vec<(u8, u8, u8)>) -> PyResult<()> {
        if qubits.len() > 8 || images.len() != 2 * qubits.len() {
            return Err(PyValueError::new_err("a gate needs the images of X and Z on each of its (at most 8) qubits"));
        }
        for (a, &q) in qubits.iter().enumerate() {
            if q >= self.sim.num_qubits {
                return Err(PyValueError::new_err(format!("qubit {q} out of range")));
            }
            if qubits[..a].contains(&q) {
                return Err(PyValueError::new_err("a gate's qubits must differ"));
            }
        }
        let images: Vec<_> = images.into_iter().map(|(phase, x, z)| crate::pecos_sparse::Image { phase: phase % 4, x, z }).collect();
        self.sim.gate(&qubits, &images);
        Ok(())
    }
    fn is_deterministic(&self, qubit: usize) -> PyResult<bool> {
        if qubit >= self.sim.num_qubits {
            return Err(PyValueError::new_err(format!("qubit {qubit} out of range")));
        }
        Ok(self.sim.is_deterministic(qubit))
    }
    /// Measures Z on `qubit`: its determined outcome, or else `outcome` (0 or 1), collapsing
    /// the state to it when `collapse`.
    #[pyo3(signature = (qubit, outcome, collapse=true))]
    fn measure(&mut self, qubit: usize, outcome: i64, collapse: bool) -> PyResult<u8> {
        if qubit >= self.sim.num_qubits {
            return Err(PyValueError::new_err(format!("qubit {qubit} out of range")));
        }
        if !self.sim.is_deterministic(qubit) && !(outcome == 0 || outcome == 1) {
            return Err(PyValueError::new_err("forced_outcome was not 0 or 1"));
        }
        Ok(self.sim.measure(qubit, outcome.clamp(0, 1) as u8, collapse))
    }
    #[getter]
    fn stabs(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        pecos_generators(py, &self.sim.stabs)
    }
    #[getter]
    fn destabs(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        pecos_generators(py, &self.sim.destabs)
    }
    #[getter]
    fn signs_minus(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut v: Vec<u32> = self.sim.signs_minus.iter().copied().collect();
        v.sort_unstable();
        Ok(pyo3::types::PySet::new(py, &v)?.into_any().unbind())
    }
    #[getter]
    fn signs_i(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut v: Vec<u32> = self.sim.signs_i.iter().copied().collect();
        v.sort_unstable();
        Ok(pyo3::types::PySet::new(py, &v)?.into_any().unbind())
    }
    /// Replaces the generators by rows (lists of sets of qubits, one per generator).
    fn set_rows(&mut self, stabs_x: Vec<Vec<u32>>, stabs_z: Vec<Vec<u32>>, destabs_x: Vec<Vec<u32>>, destabs_z: Vec<Vec<u32>>) -> PyResult<()> {
        let n = self.sim.num_qubits;
        let conv = |rows: Vec<Vec<u32>>| -> PyResult<Vec<crate::pecos_sparse::Set>> {
            if rows.len() != n || rows.iter().flatten().any(|&q| q as usize >= n) {
                return Err(PyValueError::new_err(format!("rows must be {n} sets of qubits below {n}")));
            }
            Ok(rows.into_iter().map(|r| r.into_iter().collect()).collect())
        };
        let (a, b, c, d) = (conv(stabs_x)?, conv(stabs_z)?, conv(destabs_x)?, conv(destabs_z)?);
        self.sim.set_rows(a, b, c, d);
        Ok(())
    }
    /// The signs, for `set_rows`'s callers: (minus, i) as sets of generator ids.
    fn set_signs(&mut self, minus: Vec<u32>, i: Vec<u32>) {
        self.sim.signs_minus = minus.into_iter().collect();
        self.sim.signs_i = i.into_iter().collect();
    }
    fn __copy__(&self) -> Self {
        PyPecosSparseSim { sim: self.sim.clone() }
    }
}

#[pyclass(name = "PreDecoder", module = "stabilizer_qec._core")]
pub struct PyPreDecoder {
    layout: crate::predecoder::Layout,
    net: crate::predecoder::Net,
    trainer: Option<crate::predecoder::Trainer>,
    sampler: crate::dem_sampler::DemSampler,
    /// Matching or union-find on the network's priors.
    soft: Option<crate::predecoder::Soft>,
}

fn f32_bytes(xs: &[f32]) -> Vec<u8> {
    xs.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[pymethods]
impl PyPreDecoder {
    /// `hidden`: each hidden layer's filters. `soft`: matching or union_find, to decode on the
    /// network's priors.
    #[new]
    #[pyo3(signature = (dem, hidden, kernel, seed, sectors=None, soft=None))]
    fn new(dem: &PyDem, hidden: Vec<usize>, kernel: usize, seed: u64, sectors: Option<Vec<u32>>, soft: Option<&str>) -> PyResult<Self> {
        let d = dem.flat()?;
        let layout = crate::predecoder::Layout::new(d, sectors.as_deref()).map_err(err)?;
        let mut channels = vec![layout.num_inputs()];
        channels.extend(hidden);
        channels.push(layout.num_kinds());
        let net = crate::predecoder::Net::new(&channels, kernel, seed).map_err(err)?;
        let soft = match soft {
            None => None,
            Some(kind @ ("matching" | "union_find")) => Some(crate::predecoder::Soft::new(&layout, crate::reweight::PriorDecoder::new(prior_inner(d, kind, None)?))),
            Some(other) => return Err(err(format!("decoder '{other}' is not matching or union_find"))),
        };
        Ok(PyPreDecoder { layout, net, trainer: None, sampler: crate::dem_sampler::DemSampler::new(d), soft })
    }

    #[getter]
    fn dims(&self) -> (usize, usize, usize) {
        let [t, y, x] = self.layout.dims;
        (t, y, x)
    }
    #[getter]
    fn num_sectors(&self) -> usize {
        self.layout.num_sectors
    }
    #[getter]
    fn sectors(&self) -> Vec<u32> {
        self.layout.sector.clone()
    }
    #[getter]
    fn num_detectors(&self) -> usize {
        self.layout.num_detectors
    }
    #[getter]
    fn num_observables(&self) -> usize {
        self.layout.num_observables
    }
    #[getter]
    fn num_errors(&self) -> usize {
        self.layout.parts.len()
    }
    /// Per output channel: (sector, boundary, dt, dy, dx).
    #[getter]
    fn kinds(&self) -> Vec<(u32, bool, i32, i32, i32)> {
        self.layout.kinds.iter().map(|k| (k.sector, k.boundary, k.dt, k.dy, k.dx)).collect()
    }
    /// Per voxel and kind, 1 where a fault is.
    #[getter]
    fn mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.layout.slot.iter().map(|&s| u8::from(s != u32::MAX)).collect::<Vec<u8>>())
    }
    /// Per fault the network can apply, by voxel then kind (the order of the mask's ones):
    /// (detectors, observables, probability).
    #[getter]
    fn faults(&self) -> Vec<(Vec<u32>, u64, f64)> {
        self.layout.faults.iter().map(|f| (f.detectors.clone(), f.observables, f.p)).collect()
    }
    #[getter]
    fn channels(&self) -> Vec<usize> {
        self.net.channels.clone()
    }
    #[getter]
    fn kernel(&self) -> usize {
        self.net.kernel
    }
    /// The parameters, f32 little-endian: per layer its weights (kernel offset, input channel,
    /// output channel), then its biases.
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &f32_bytes(&self.net.params))
    }
    #[setter]
    fn set_params(&mut self, raw: &[u8]) -> PyResult<()> {
        if raw.len() != 4 * self.net.params.len() {
            return Err(err(format!("{} bytes is not {} parameters", raw.len(), self.net.params.len())));
        }
        for (p, b) in self.net.params.iter_mut().zip(raw.chunks_exact(4)) {
            *p = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        }
        Ok(())
    }

    /// The inputs (f32, shot by voxel by channel) of packed detection events, and with the
    /// faults that fired (packed, the model's order), the labels (f32, shot by voxel by kind).
    #[pyo3(signature = (dets, shots, errors=None))]
    fn encode<'py>(&self, py: Python<'py>, dets: &[u8], shots: usize, errors: Option<&[u8]>) -> PyResult<(Bound<'py, PyBytes>, Option<Bound<'py, PyBytes>>)> {
        let l = &self.layout;
        let (sd, se) = (l.num_detectors.div_ceil(8), l.parts.len().div_ceil(8));
        check_rows(dets.len(), sd, shots, l.num_detectors, "detectors")?;
        let (nx, ny) = (l.num_voxels() * l.num_inputs(), l.num_voxels() * l.num_kinds());
        let mut x = vec![0.0f32; shots * nx];
        for s in 0..shots {
            l.encode(&dets[s * sd..(s + 1) * sd], &mut x[s * nx..(s + 1) * nx]);
        }
        let y = match errors {
            None => None,
            Some(e) => {
                check_rows(e.len(), se, shots, l.parts.len(), "errors")?;
                let mut y = vec![0.0f32; shots * ny];
                for s in 0..shots {
                    l.labels(&e[s * se..(s + 1) * se], &mut y[s * ny..(s + 1) * ny]);
                }
                Some(PyBytes::new(py, &f32_bytes(&y)))
            }
        };
        Ok((PyBytes::new(py, &f32_bytes(&x)), y))
    }

    /// The shots training draws: `shots` of the model's, the sampler's batches from `first` on
    /// (packed detection events, packed faults that fired).
    fn sample<'py>(&self, py: Python<'py>, shots: usize, seed: u64, first: u64, threads: usize) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>) {
        let (d, _, e) = py.detach(|| self.sampler.sample_seeded(seed, first, shots, threads, true));
        (PyBytes::new(py, &d), PyBytes::new(py, &e))
    }

    /// The number of optimizer steps taken.
    #[getter]
    fn steps(&self) -> u64 {
        self.trainer.as_ref().map_or(0, |t| t.step)
    }

    /// The mean loss of packed shots and the faults that fired, and its gradient (f32 bytes).
    fn loss_and_grad<'py>(&self, py: Python<'py>, dets: &[u8], errors: &[u8], shots: usize, threads: usize) -> PyResult<(f64, Bound<'py, PyBytes>)> {
        let l = &self.layout;
        check_rows(dets.len(), l.num_detectors.div_ceil(8), shots, l.num_detectors, "detectors")?;
        check_rows(errors.len(), l.parts.len().div_ceil(8), shots, l.parts.len(), "errors")?;
        let (loss, grad) = py.detach(|| crate::predecoder::loss_and_grad(l, &self.net, dets, errors, shots, threads));
        Ok((loss, PyBytes::new(py, &f32_bytes(&grad))))
    }

    /// `steps` steps of `batch` fresh shots each; the optimizer's state carries over between
    /// calls with the same optimizer. Returns each step's mean loss.
    #[pyo3(signature = (steps, batch, optimizer, lr, warmup, weight_decay, seed, threads))]
    #[allow(clippy::too_many_arguments)]
    fn train(&mut self, py: Python<'_>, steps: usize, batch: usize, optimizer: &str, lr: f32, warmup: usize, weight_decay: f32, seed: u64, threads: usize) -> PyResult<Vec<f64>> {
        use crate::predecoder::{Optimizer, Trainer};
        let opt = match optimizer {
            "lion" => Optimizer::Lion { weight_decay },
            "adam" => Optimizer::Adam,
            other => return Err(err(format!("optimizer '{other}' is not lion or adam"))),
        };
        if self.trainer.as_ref().map(|t| t.optimizer) != Some(opt) {
            self.trainer = Some(Trainer::new(opt, self.net.params.len()));
        }
        let PyPreDecoder { layout, net, trainer, sampler, .. } = self;
        let trainer = trainer.as_mut().unwrap();
        Ok(py.detach(|| crate::predecoder::train(layout, net, trainer, sampler, steps, batch.max(1), lr, warmup, seed, threads)))
    }

    /// Per shot, the observables of the faults applied and of the soft decoder's correction of
    /// the rest on the network's priors (u64 each), and 1 where no correction explains the shot.
    fn decode_soft<'py>(&self, py: Python<'py>, packed: &[u8], shots: usize, threshold: f32, threads: usize) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let l = &self.layout;
        check_rows(packed.len(), l.num_detectors.div_ceil(8), shots, l.num_detectors, "detectors")?;
        let soft = self.soft.as_ref().ok_or_else(|| err("this pre-decoder has no soft decoder".into()))?;
        let out = py.detach(|| crate::predecoder::decode_soft(l, &self.net, soft, packed, shots, threshold, threads));
        let failed: Vec<u8> = out.iter().map(|o| u8::from(o.is_none())).collect();
        Ok((PyBytes::new(py, &le_u64(out.iter().map(|o| o.unwrap_or(0)))), PyBytes::new(py, &failed)))
    }

    /// Per shot: the detection events left (packed), the observables of the faults applied (u64
    /// each), how many were applied (u32 each).
    fn predecode<'py>(&self, py: Python<'py>, packed: &[u8], shots: usize, threshold: f32, threads: usize) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let l = &self.layout;
        check_rows(packed.len(), l.num_detectors.div_ceil(8), shots, l.num_detectors, "detectors")?;
        let (left, obs, applied) = py.detach(|| crate::predecoder::predecode(l, &self.net, packed, shots, threshold, threads));
        let applied: Vec<u8> = applied.iter().flat_map(|a| a.to_le_bytes()).collect();
        Ok((PyBytes::new(py, &left), PyBytes::new(py, &le_u64(obs)), PyBytes::new(py, &applied)))
    }
}

/// Leakage-aware decoding from heralds around a decoder that weighs faults by their priors.
#[pyclass(name = "DemLeakageAware", module = "stabilizer_qec._core")]
pub struct PyDemLeakageAware {
    dec: crate::leakage::LeakageAware,
    #[pyo3(get)]
    num_detectors: usize,
    #[pyo3(get)]
    num_observables: usize,
    /// The triggers' faults that are not the decoder's (a fault of the leaked circuit's model
    /// with no match in the decoder's).
    #[pyo3(get)]
    unmatched: usize,
}

#[pymethods]
impl PyDemLeakageAware {
    /// `triggers`: per trigger, its faults as (detectors, observables, probability given the
    /// trigger). For matching and union-find each must be an edge (one or two detectors); for
    /// the BP decoders, one of the model's faults.
    #[new]
    #[pyo3(signature = (dem, kind, inner, triggers))]
    fn new(dem: &PyDem, kind: &str, inner: Option<&Bound<'_, PyAny>>, triggers: Vec<Vec<(Vec<u32>, u64, f64)>>) -> PyResult<Self> {
        let d = dem.flat()?;
        let dec = crate::reweight::PriorDecoder::new(prior_inner(d, kind, inner)?);
        let n = d.num_detectors as u32;
        let mut index: std::collections::HashMap<(Vec<u32>, u64), u32> = std::collections::HashMap::new();
        if !matches!(kind, "matching" | "union_find") {
            for (k, m) in d.mechanisms.iter().enumerate() {
                let mut dets = m.detectors.clone();
                dets.sort_unstable();
                index.entry((dets, m.observables)).or_insert(k as u32);
            }
        }
        let mut unmatched = 0usize;
        let mut resolved = Vec::with_capacity(triggers.len());
        for faults in triggers {
            let mut q_of: std::collections::BTreeMap<u32, f64> = std::collections::BTreeMap::new();
            for (mut dets, obs, q) in faults {
                dets.sort_unstable();
                let fault = match (kind, dets.as_slice()) {
                    ("matching" | "union_find", [a]) => dec.edge(*a, n),
                    ("matching" | "union_find", [a, b]) => dec.edge(*a, *b),
                    ("matching" | "union_find", _) => None,
                    _ => index.get(&(dets, obs)).copied(),
                };
                match fault {
                    Some(f) => {
                        let p = q_of.entry(f).or_insert(0.0);
                        *p = *p * (1.0 - q) + q * (1.0 - *p);
                    }
                    None => unmatched += 1,
                }
            }
            resolved.push(q_of.into_iter().collect());
        }
        let dec = crate::leakage::LeakageAware::new(dec, resolved).map_err(err)?;
        Ok(PyDemLeakageAware { dec, num_detectors: d.num_detectors, num_observables: d.num_observables, unmatched })
    }

    /// Per trigger, how many of the decoder's faults it raises.
    #[getter]
    fn trigger_sizes(&self) -> Vec<usize> {
        self.dec.trigger_sizes()
    }

    /// (observables as u64 per shot, one byte per shot: 1 where no correction explains it; the
    /// priors each shot raised, u32 each). `triggers`: b8 rows, one bit per trigger.
    #[pyo3(signature = (packed, shots, threads, triggers=None))]
    fn decode_batch<'py>(&self, py: Python<'py>, packed: &[u8], shots: usize, threads: usize, triggers: Option<&[u8]>) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let nd = self.num_detectors;
        check_rows(packed.len(), nd.div_ceil(8), shots, nd, "detectors")?;
        let nt = self.dec.num_triggers();
        if let Some(t) = triggers {
            check_rows(t.len(), nt.div_ceil(8), shots, nt, "triggers")?;
        }
        let dec = &self.dec;
        let out = py.detach(|| crate::batch::leakage_aware_shots(dec, packed, nd, triggers, shots, threads));
        let failed: Vec<u8> = out.iter().map(|o| u8::from(o.0.is_none())).collect();
        let raised: Vec<u8> = out.iter().flat_map(|o| o.1.to_le_bytes()).collect();
        Ok((PyBytes::new(py, &le_u64(out.iter().map(|o| o.0.unwrap_or(0)))), PyBytes::new(py, &failed), PyBytes::new(py, &raised)))
    }
}

#[pyclass(name = "DemArgumentReweighting", module = "stabilizer_qec._core")]
pub struct PyDemArgumentReweighting {
    dec: crate::reweight::ArgumentReweighting,
    #[pyo3(get)]
    num_detectors: usize,
    #[pyo3(get)]
    num_observables: usize,
}

#[pymethods]
impl PyDemArgumentReweighting {
    /// `inner`: the BP decoder's core (a `DemBpOsd`, `DemBpLsd` or `DemRelay`), for those kinds.
    #[new]
    #[pyo3(signature = (dem, kind, inner, b, test, criterion))]
    fn new(dem: &PyDem, kind: &str, inner: Option<&Bound<'_, PyAny>>, b: f64, test: &str, criterion: &str) -> PyResult<Self> {
        use crate::reweight::{Criterion, Test};
        let d = dem.flat()?;
        let inner = prior_inner(&d, kind, inner)?;
        let test = match test {
            "ratio" => Test::Ratio,
            "exact_ratio" => Test::ExactRatio,
            "gap" => Test::Gap,
            other => return Err(err(format!("test '{other}' is not ratio, exact_ratio or gap"))),
        };
        let criterion = match criterion {
            "pec" => Criterion::Physical,
            "2r-lec" => Criterion::TwoRound,
            "3r-lec" => Criterion::ThreeRound,
            other => return Err(err(format!("criterion '{other}' is not pec, 2r-lec or 3r-lec"))),
        };
        let dec = crate::reweight::ArgumentReweighting::new(inner, b, test, criterion).map_err(err)?;
        Ok(PyDemArgumentReweighting { dec, num_detectors: d.num_detectors, num_observables: d.num_observables })
    }

    /// (observables as u64 per shot, one byte per shot: 1 kept, 0 rejected, 2 no correction;
    /// one byte per shot: the decodes run).
    fn decode_batch<'py>(&self, py: Python<'py>, packed: &[u8], shots: usize, threads: usize) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let nd = self.num_detectors;
        check_rows(packed.len(), nd.div_ceil(8), shots, nd, "detectors")?;
        let dec = &self.dec;
        let out = py.detach(|| crate::batch::reweight_shots(dec, packed, nd, shots, threads));
        let flags: Vec<u8> = out.iter().map(|o| if o.failed { 2 } else { u8::from(o.accepted) }).collect();
        let rounds: Vec<u8> = out.iter().map(|o| o.rounds).collect();
        Ok((PyBytes::new(py, &le_u64(out.iter().map(|o| o.observables))), PyBytes::new(py, &flags), PyBytes::new(py, &rounds)))
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

fn schedule(name: &str) -> PyResult<crate::bp::Schedule> {
    match name {
        "parallel" => Ok(crate::bp::Schedule::Parallel),
        "serial" => Ok(crate::bp::Schedule::Serial),
        "serial_relative" => Ok(crate::bp::Schedule::SerialRelative),
        other => Err(err(format!("BP schedule '{other}' is not parallel, serial or serial_relative"))),
    }
}

fn checked_syndrome(syndrome: &[u8], num_checks: usize) -> PyResult<()> {
    if syndrome.len() != num_checks {
        return Err(err(format!("{} syndrome bits for {} checks", syndrome.len(), num_checks)));
    }
    Ok(())
}

fn checked_order(order: &[u32], n: usize) -> PyResult<()> {
    if order.len() != n || order.iter().any(|&v| v as usize >= n) {
        return Err(err(format!("a serial schedule order has {n} entries, each below {n}")));
    }
    Ok(())
}

/// Belief propagation on a check matrix given by its columns, as `ldpc`'s `BpDecoder`: every
/// schedule, and the state `ldpc` keeps from decode to decode (the last decision and
/// posteriors, the serial order and its generator).
#[pyclass(name = "Bp", module = "stabilizer_qec._core")]
pub struct PyBp {
    bp: crate::bp::Bp,
    method: crate::bp::Method,
    max_iter: usize,
    num_checks: usize,
    schedule: crate::bp::Schedule,
    serial: crate::bp::SerialState,
    work: crate::bp::BpWork,
    /// The GPU's pacing of this decoder's batches, learned across calls.
    pace: crate::lanes::PaceHint,
}

#[pymethods]
impl PyBp {
    #[new]
    fn new(num_checks: usize, columns: Vec<Vec<u32>>, priors: Vec<f64>, max_iter: usize, method: &str, scale: f64) -> PyResult<Self> {
        let mut bp = crate::bp::Bp::new(num_checks, &columns, &priors).map_err(err)?;
        bp.zero_shortcut = false;
        let (work, serial) = (bp.work(), crate::bp::SerialState::new(columns.len()));
        Ok(PyBp { bp, method: bp_method(method, scale)?, max_iter, num_checks, schedule: crate::bp::Schedule::Parallel, serial, work, pace: crate::lanes::PaceHint::default() })
    }

    fn set_priors(&mut self, priors: Vec<f64>) -> PyResult<()> {
        self.bp.set_priors(&priors).map_err(err)
    }

    fn set_max_iter(&mut self, max_iter: usize) {
        self.max_iter = max_iter;
    }

    fn set_method(&mut self, method: &str, scale: f64) -> PyResult<()> {
        self.method = bp_method(method, scale)?;
        Ok(())
    }

    fn set_schedule(&mut self, name: &str) -> PyResult<()> {
        self.schedule = schedule(name)?;
        Ok(())
    }

    fn set_order(&mut self, order: Vec<u32>) -> PyResult<()> {
        checked_order(&order, self.bp.num_vars)?;
        self.serial.order = order;
        Ok(())
    }

    fn order(&self) -> Vec<u32> {
        self.serial.order.clone()
    }

    fn set_random(&mut self, random: bool) {
        self.serial.random = random;
    }

    fn seed(&mut self, seed: u32) {
        self.serial.seed(seed);
    }

    /// (hard decision, posterior log-likelihood ratios, converged, iterations).
    fn decode(&mut self, syndrome: Vec<u8>) -> PyResult<(Vec<u32>, Vec<f64>, bool, usize)> {
        checked_syndrome(&syndrome, self.num_checks)?;
        let out = self.bp.decode_scheduled(&syndrome, self.method, self.max_iter, self.schedule, &mut self.serial, &mut self.work);
        Ok((bits(&self.work.hard), self.work.llr.clone(), out.converged, out.iterations))
    }

    /// Min-sum in single precision, the parallel schedule, a batch at once: `syndromes` one
    /// byte per check per shot, back to back; on "cpu-f32" (`threads` wide) or "gpu". Returns
    /// (hard decisions as b8 rows, one byte per shot set where BP converged, iterations).
    fn decode_batch32<'py>(&self, py: Python<'py>, syndromes: &[u8], shots: usize, device: &str, threads: usize) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Vec<u32>)> {
        if syndromes.len() != shots * self.num_checks {
            return Err(err(format!("{} syndrome bytes for {shots} shots of {} checks", syndromes.len(), self.num_checks)));
        }
        let scale = match self.method {
            crate::bp::Method::MinSum { scale } => scale as f32,
            crate::bp::Method::ProductSum => return Err(err("single-precision BP is min-sum only; set bp_method to minimum_sum".to_string())),
        };
        if self.schedule != crate::bp::Schedule::Parallel {
            return Err(err("single-precision BP runs the parallel schedule only".to_string()));
        }
        let bp32 = crate::bp32::Bp32::from_bp(&self.bp);
        let max_iter = self.max_iter;
        let out = match device {
            "cpu-f32" => py.detach(|| bp32.decode_batch(syndromes, shots, scale, max_iter, threads)),
            "gpu" => {
                #[cfg(feature = "gpu")]
                {
                    let pace = &self.pace;
                    py.detach(|| crate::gpu::bp_decode_batch(&bp32, syndromes, shots, scale, max_iter, pace)).map_err(err)?
                }
                #[cfg(not(feature = "gpu"))]
                {
                    return Err(err("device 'gpu' needs the package built with its gpu feature (pip install stabilizer-qec[gpu])".to_string()));
                }
            }
            other => return Err(err(format!("device must be 'cpu-f32' or 'gpu', not {other:?}"))),
        };
        Ok((PyBytes::new(py, &out.hard), PyBytes::new(py, &out.converged), out.iterations))
    }

    /// Soft-information serial min-sum: (hard decision, posteriors, converged, iterations, the
    /// soft syndrome after decoding).
    #[allow(clippy::type_complexity)]
    fn decode_soft(&mut self, soft: Vec<f64>, cutoff: f64, sigma: f64, scale: f64, seed: u32) -> PyResult<(Vec<u32>, Vec<f64>, bool, usize, Vec<f64>)> {
        if soft.len() != self.num_checks {
            return Err(err(format!("{} soft syndrome values for {} checks", soft.len(), self.num_checks)));
        }
        let (out, soft) = self.bp.decode_soft_serial(&soft, cutoff, sigma, scale, self.max_iter, &mut self.serial, seed, &mut self.work);
        Ok((bits(&self.work.hard), self.work.llr.clone(), out.converged, out.iterations, soft))
    }
}

/// BP+OSD on a check matrix given by its columns, as `ldpc`'s `BpOsdDecoder`.
#[pyclass(name = "BpOsd", module = "stabilizer_qec._core")]
pub struct PyBpOsd {
    dec: BpOsd,
    num_checks: usize,
    work: crate::osd::BpOsdWork,
}

#[pymethods]
impl PyBpOsd {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(num_checks: usize, columns: Vec<Vec<u32>>, priors: Vec<f64>, max_iter: usize, method: &str, scale: f64, osd: &str, order: usize) -> PyResult<Self> {
        let mut dec = BpOsd::new(num_checks, columns, &priors, bp_method(method, scale)?, max_iter, osd_method(osd, order)?).map_err(err)?;
        // As ldpc's BpDecoder: BP runs on a zero syndrome too (a received codeword's), its
        // iterations and posteriors that run's. (BpOsdDecoder answers a zero syndrome itself.)
        dec.bp.zero_shortcut = false;
        let work = dec.work();
        Ok(PyBpOsd { dec, num_checks, work })
    }

    fn set_priors(&mut self, priors: Vec<f64>) -> PyResult<()> {
        self.dec.set_priors(&priors).map_err(err)
    }

    fn set_max_iter(&mut self, max_iter: usize) {
        self.dec.max_iter = max_iter;
    }

    fn set_method(&mut self, method: &str, scale: f64) -> PyResult<()> {
        self.dec.method = bp_method(method, scale)?;
        Ok(())
    }

    fn set_schedule(&mut self, name: &str) -> PyResult<()> {
        self.dec.schedule = schedule(name)?;
        Ok(())
    }

    fn set_order(&mut self, order: Vec<u32>) -> PyResult<()> {
        checked_order(&order, self.dec.bp.num_vars)?;
        self.work.serial.order = order;
        Ok(())
    }

    fn order(&self) -> Vec<u32> {
        self.work.serial.order.clone()
    }

    fn set_random(&mut self, random: bool) {
        self.work.serial.random = random;
    }

    fn seed(&mut self, seed: u32) {
        self.work.serial.seed(seed);
    }

    /// The OSD order now (the candidates stay those set up at construction, as in `ldpc`).
    fn set_osd_order(&mut self, order: usize) {
        self.dec.order = order;
    }

    /// (correction, BP converged, iterations, BP's decision, posteriors, OSD-0's solution).
    #[allow(clippy::type_complexity)]
    fn decode(&mut self, syndrome: Vec<u8>) -> PyResult<(Vec<u32>, bool, usize, Vec<u32>, Vec<f64>, Vec<u32>)> {
        checked_syndrome(&syndrome, self.num_checks)?;
        let out = self.dec.decode(&syndrome, &mut self.work);
        let w = &self.work;
        Ok((bits(&w.correction), out.converged, out.iterations, bits(&w.bp.hard), w.bp.llr.clone(), bits(&w.osd0)))
    }

    /// OSD alone on given posteriors: (correction, OSD-0's solution).
    fn osd_on(&mut self, syndrome: Vec<u8>, llr: Vec<f64>) -> PyResult<(Vec<u32>, Vec<u32>)> {
        checked_syndrome(&syndrome, self.num_checks)?;
        if llr.len() != self.work.correction.len() {
            return Err(err(format!("{} posteriors for {} columns", llr.len(), self.work.correction.len())));
        }
        self.dec.osd_on(&syndrome, &llr, &mut self.work);
        Ok((bits(&self.work.correction), bits(&self.work.osd0)))
    }
}

/// BP+LSD on a check matrix given by its columns, as `ldpc`'s `BpLsdDecoder`.
#[pyclass(name = "BpLsd", module = "stabilizer_qec._core")]
pub struct PyBpLsd {
    dec: BpLsd,
    num_checks: usize,
    work: crate::lsd::BpLsdWork,
}

#[pymethods]
impl PyBpLsd {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(num_checks: usize, columns: Vec<Vec<u32>>, priors: Vec<f64>, max_iter: usize, method: &str, scale: f64, lsd: &str, order: usize, bits_per_step: usize, always_run: bool) -> PyResult<Self> {
        let mut dec = BpLsd::new(num_checks, columns, &priors, bp_method(method, scale)?, max_iter, osd_method(lsd, order)?, bits_per_step, always_run).map_err(err)?;
        // As ldpc's BpLsdDecoder: BP runs on every syndrome, a zero one included.
        dec.zero_shortcut = false;
        dec.bp.zero_shortcut = false;
        let work = dec.work();
        Ok(PyBpLsd { dec, num_checks, work })
    }

    fn set_priors(&mut self, priors: Vec<f64>) -> PyResult<()> {
        self.dec.bp.set_priors(&priors).map_err(err)
    }

    fn set_max_iter(&mut self, max_iter: usize) {
        self.dec.max_iter = max_iter;
    }

    fn set_method(&mut self, method: &str, scale: f64) -> PyResult<()> {
        self.dec.method = bp_method(method, scale)?;
        Ok(())
    }

    fn set_schedule(&mut self, name: &str) -> PyResult<()> {
        self.dec.schedule = schedule(name)?;
        Ok(())
    }

    fn set_order(&mut self, order: Vec<u32>) -> PyResult<()> {
        checked_order(&order, self.dec.bp.num_vars)?;
        self.work.serial.order = order;
        Ok(())
    }

    fn order(&self) -> Vec<u32> {
        self.work.serial.order.clone()
    }

    fn set_random(&mut self, random: bool) {
        self.work.serial.random = random;
    }

    fn seed(&mut self, seed: u32) {
        self.work.serial.seed(seed);
    }

    /// The LSD method and order now (`ldpc`'s `lsd_method` and `lsd_order` setters, which the
    /// next decode reads); `"osd_off"` is `ldpc`'s `OSD_OFF`.
    fn set_lsd(&mut self, lsd: &str, order: usize) -> PyResult<()> {
        let off = lsd == "osd_off";
        self.dec.lsd = if off { OsdMethod::Exhaustive(order) } else { osd_method(lsd, order)? };
        self.dec.off = off;
        Ok(())
    }

    fn set_do_stats(&mut self, on: bool) {
        self.work.do_stats = on;
    }

    #[getter]
    fn do_stats(&self) -> bool {
        self.work.do_stats
    }

    fn reset_stats(&mut self) {
        self.work.stats.clear();
    }

    /// The last decode's statistics, laid out as `ldpc`'s `statistics` dict (less the fields
    /// a caller sets).
    fn statistics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let st = &self.work.stats;
        let clusters = PyDict::new(py);
        for (id, c) in &st.clusters {
            let d = PyDict::new(py);
            d.set_item("final_bit_count", c.final_bit_count)?;
            d.set_item("undergone_growth_steps", c.undergone_growth_steps)?;
            d.set_item("nr_merges", c.nr_merges)?;
            d.set_item("active", c.active)?;
            d.set_item("size_history", c.size_history.clone())?;
            d.set_item("got_valid_in_timestep", c.got_valid_in_timestep)?;
            d.set_item("got_inactive_in_timestep", c.got_inactive_in_timestep)?;
            d.set_item("absorbed_by_cluster", c.absorbed_by_cluster)?;
            d.set_item("nr_of_non_zero_check_matrix_entries", c.nr_of_non_zero_check_matrix_entries)?;
            d.set_item("cluster_pcm_sparsity", c.cluster_pcm_sparsity)?;
            d.set_item("solution", bits(&c.solution))?;
            d.set_item("final_bits", c.final_bits.clone())?;
            clusters.set_item(id, d)?;
        }
        let history = PyDict::new(py);
        for (t, m) in &st.history {
            let d = PyDict::new(py);
            for (id, b) in m {
                d.set_item(id, b.clone())?;
            }
            history.set_item(t, d)?;
        }
        let out = PyDict::new(py);
        out.set_item("individual_cluster_stats", clusters)?;
        out.set_item("global_timestep_bit_history", history)?;
        out.set_item("elapsed_time", st.elapsed_micros)?;
        out.set_item("lsd_order", st.lsd_order)?;
        out.set_item("lsd_method", st.lsd_method)?;
        out.set_item("bit_llrs", st.bit_llrs.clone())?;
        out.set_item("syndrome", bits(&st.syndrome))?;
        Ok(out)
    }

    /// LSD alone on `weights`: (correction, solved, ties, final clusters' bits when `record`).
    #[pyo3(signature = (syndrome, weights, record=false))]
    #[allow(clippy::type_complexity)]
    fn decode_lsd(&mut self, syndrome: Vec<u8>, weights: Vec<f64>, record: bool) -> PyResult<(Vec<u32>, bool, u32, Vec<(u32, Vec<u32>)>)> {
        checked_syndrome(&syndrome, self.num_checks)?;
        if weights.len() != self.dec.bp.num_vars {
            return Err(err(format!("{} bit weights for {} bits", weights.len(), self.dec.bp.num_vars)));
        }
        self.work.record = record;
        let solved = self.dec.lsd_only(&syndrome, &weights, &mut self.work);
        let w = &mut self.work;
        Ok((bits(&w.correction), solved, w.ties, std::mem::take(&mut w.cluster_bits)))
    }

    /// (correction, BP converged, iterations, solved, ties, each final cluster's bits in
    /// set order by cluster id when `record`, BP's decision, posteriors).
    #[pyo3(signature = (syndrome, record=false))]
    #[allow(clippy::type_complexity)]
    fn decode(&mut self, syndrome: Vec<u8>, record: bool) -> PyResult<(Vec<u32>, bool, usize, bool, u32, Vec<(u32, Vec<u32>)>, Vec<u32>, Vec<f64>)> {
        checked_syndrome(&syndrome, self.num_checks)?;
        self.work.record = record;
        let out = self.dec.decode(&syndrome, &mut self.work);
        let w = &mut self.work;
        Ok((bits(&w.correction), out.converged, out.iterations, out.solved, w.ties, std::mem::take(&mut w.cluster_bits), bits(&w.bp.hard), w.bp.llr.clone()))
    }
}

/// Bit flipping on a check matrix given by its columns, as `ldpc`'s `FlipDecoder`.
#[pyclass(name = "Flip", module = "stabilizer_qec._core")]
pub struct PyFlip {
    dec: crate::flip::Flip,
    num_checks: usize,
    n: usize,
}

#[pymethods]
impl PyFlip {
    #[new]
    fn new(num_checks: usize, columns: Vec<Vec<u32>>, max_iter: usize, pfreq: usize, seed: u32) -> Self {
        let n = columns.len();
        PyFlip { dec: crate::flip::Flip::new(num_checks, columns, max_iter, pfreq, seed), num_checks, n }
    }

    /// (decoding, converged, iterations).
    fn decode(&mut self, syndrome: Vec<u8>) -> PyResult<(Vec<u32>, bool, usize)> {
        checked_syndrome(&syndrome, self.num_checks)?;
        let mut d = vec![0u8; self.n];
        let out = self.dec.decode(&syndrome, &mut d);
        Ok((bits(&d), out.converged, out.iterations))
    }
}

/// `ldpc`'s union-find decoder on a check matrix given by its columns.
#[pyclass(name = "LdpcUf", module = "stabilizer_qec._core")]
pub struct PyLdpcUf {
    dec: crate::ldpc_uf::LdpcUf,
}

#[pymethods]
impl PyLdpcUf {
    #[new]
    fn new(num_checks: usize, columns: Vec<Vec<u32>>) -> PyResult<Self> {
        Ok(PyLdpcUf { dec: crate::ldpc_uf::LdpcUf::new(num_checks, columns).map_err(PyRuntimeError::new_err)? })
    }

    #[getter]
    fn max_degree_2(&self) -> bool {
        self.dec.max_degree_2()
    }

    /// (decoding, solved, ties, stuck): matrix inversion, or peeling when `peel`.
    #[pyo3(signature = (syndrome, weights, per_step, peel))]
    fn decode(&self, syndrome: Vec<u8>, weights: Option<Vec<f64>>, per_step: usize, peel: bool) -> PyResult<(Vec<u32>, bool, u32, u32)> {
        let out = self.dec.decode(&syndrome, weights.as_deref(), per_step, peel).map_err(PyValueError::new_err)?;
        Ok((bits(&out.decoding), out.solved, out.ties, out.stuck))
    }
}

fn three(v: Vec<Vec<f64>>, what: &str) -> PyResult<[Vec<f64>; 3]> {
    <[Vec<f64>; 3]>::try_from(v).map_err(|_| PyValueError::new_err(format!("{what} needs three rows (X, Y, Z)")))
}

/// Memory belief propagation over GF(4), as `ldpc`'s `mbp_decoder`; its state carries over.
#[pyclass(name = "Mbp", module = "stabilizer_qec._core")]
pub struct PyMbp {
    dec: crate::mbp::Mbp,
}

#[pymethods]
impl PyMbp {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(num_checks: usize, num_qubits: usize, entries: Vec<(usize, usize, u8)>, channel: Vec<Vec<f64>>, alpha: Vec<Vec<f64>>, beta: f64, gamma: f64, min_sum: bool, max_iter: usize) -> PyResult<Self> {
        let dec = crate::mbp::Mbp::new(num_checks, num_qubits, &entries, three(channel, "the channel")?, three(alpha, "alpha")?, beta, gamma, min_sum, max_iter).map_err(PyValueError::new_err)?;
        Ok(PyMbp { dec })
    }

    /// (decoding over GF(4), converged, iterations, log-likelihood ratios).
    fn decode(&mut self, syndrome: Vec<u8>) -> PyResult<(Vec<u32>, bool, usize, Vec<Vec<f64>>)> {
        checked_syndrome(&syndrome, self.dec.num_checks())?;
        let d = self.dec.decode(&syndrome).iter().map(|&x| x as u32).collect();
        Ok((d, self.dec.converge, self.dec.iterations, self.dec.log_prob_ratios.to_vec()))
    }

    fn set_alpha(&mut self, alpha: Vec<Vec<f64>>) -> PyResult<()> {
        let alpha = three(alpha, "alpha")?;
        if alpha.iter().any(|v| v.len() != self.dec.num_qubits()) {
            return Err(PyValueError::new_err("alpha needs a value per qubit"));
        }
        self.dec.alpha = alpha;
        Ok(())
    }

    fn pauli_syndrome(&self, error: Vec<u8>) -> PyResult<Vec<u32>> {
        if error.len() != self.dec.num_qubits() {
            return Err(PyValueError::new_err(format!("the error needs {} entries", self.dec.num_qubits())));
        }
        Ok(bits(&self.dec.pauli_syndrome(&error)))
    }
}

/// One row of a batch: its correction, the checks that sets off, its posteriors (if asked
/// for), how it went, and its log lines (if asked for).
struct RelayRow {
    correction: Vec<u8>,
    decoded: Vec<u8>,
    posteriors: Vec<f64>,
    outcome: crate::relay::RelayOutcome,
    log: Vec<String>,
}

/// One of IBM's decoders in one number type, behind a trait object.
trait DynRelay: Send + Sync {
    fn run(&mut self, syndrome: &[u8], relay: bool) -> crate::relay::RelayOutcome;
    fn result(&self) -> (Vec<u32>, Vec<u32>, Vec<f64>);
    fn quality(&self, error: &[u8]) -> f64;
    fn log_lines(&self) -> Vec<String>;
    fn boxed_clone(&self) -> Box<dyn DynRelay>;
    /// `count` rows of `rows` (one byte per check each): in turn, the state going on from row
    /// to row; or with `fresh`, each from the state as it is now (left so), on `threads`
    /// threads, each thread with one copy of the state that it resets before each row.
    #[allow(clippy::too_many_arguments)]
    fn batch(&mut self, rows: &[u8], count: usize, relay: bool, fresh: bool, threads: usize, posteriors: bool, logs: bool) -> Vec<RelayRow>;
}

struct RelayEngineN<N: crate::relay_num::RelayNum> {
    core: std::sync::Arc<crate::relay::RelayCore<N>>,
    work: crate::relay::RelayCoreWork<N>,
}

fn relay_row<N: crate::relay_num::RelayNum>(core: &crate::relay::RelayCore<N>, work: &mut crate::relay::RelayCoreWork<N>, syndrome: &[u8], relay: bool, posteriors: bool, logs: bool) -> RelayRow {
    let outcome = if relay { core.decode(syndrome, work) } else { core.decode_min_sum(syndrome, work) };
    RelayRow {
        correction: work.correction.clone(),
        decoded: work.decoded_detectors.clone(),
        posteriors: if posteriors { work.posterior_ratios.clone() } else { Vec::new() },
        outcome,
        log: if logs { work.log_lines() } else { Vec::new() },
    }
}

impl<N: crate::relay_num::RelayNum> DynRelay for RelayEngineN<N> {
    fn run(&mut self, syndrome: &[u8], relay: bool) -> crate::relay::RelayOutcome {
        if relay {
            self.core.decode(syndrome, &mut self.work)
        } else {
            self.core.decode_min_sum(syndrome, &mut self.work)
        }
    }
    fn batch(&mut self, rows: &[u8], count: usize, relay: bool, fresh: bool, threads: usize, posteriors: bool, logs: bool) -> Vec<RelayRow> {
        let m = self.core.num_checks;
        if !fresh {
            let (core, work) = (&*self.core, &mut self.work);
            return (0..count).map(|i| relay_row(core, &mut *work, &rows[i * m..(i + 1) * m], relay, posteriors, logs)).collect();
        }
        let (core, start) = (&*self.core, &self.work);
        crate::parallel::parallel_dynamic(count, threads, || start.clone(), |work: &mut crate::relay::RelayCoreWork<N>, i: usize| {
            work.copy_from(start);
            relay_row(core, work, &rows[i * m..(i + 1) * m], relay, posteriors, logs)
        })
    }
    fn result(&self) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
        (bits(&self.work.correction), bits(&self.work.decoded_detectors), self.work.posterior_ratios.clone())
    }
    fn quality(&self, error: &[u8]) -> f64 {
        self.core.quality(error)
    }
    fn log_lines(&self) -> Vec<String> {
        self.work.log_lines()
    }
    fn boxed_clone(&self) -> Box<dyn DynRelay> {
        Box::new(RelayEngineN { core: self.core.clone(), work: self.work.clone() })
    }
}

fn relay_engine<N: crate::relay_num::RelayNum>(num_checks: usize, columns: &[Vec<u32>], values: Option<&[Vec<u8>]>, priors: &[f64], config: RelayConfig) -> PyResult<Box<dyn DynRelay>> {
    let core = crate::relay::RelayCore::<N>::with_values(num_checks, columns, values, priors, config).map_err(err)?;
    let work = core.work();
    Ok(Box::new(RelayEngineN { core: std::sync::Arc::new(core), work }))
}

/// One decode's result: (correction, checks it sets off, posteriors, converged, iterations,
/// iteration limit, weight).
type RelayResult = (Vec<u32>, Vec<u32>, Vec<f64>, bool, usize, usize, f64);

/// IBM's `relay_bp` decoders (`RelayDecoder*` with `relay`, `MinSumBPDecoder*` without) in the
/// number type `kind` ("f32", "f64", "i8", "i16", "i32", "i64", or "fixed" with `frac_bits`).
/// Its state, random memory strengths included, continues from decode to decode.
#[pyclass(name = "RelayEngine", module = "stabilizer_qec._core")]
pub struct PyRelayEngine {
    eng: Box<dyn DynRelay>,
    relay: bool,
    num_checks: usize,
    num_vars: usize,
}

impl PyRelayEngine {
    fn one(&mut self, syndrome: &[u8]) -> RelayResult {
        let out = self.eng.run(syndrome, self.relay);
        let (c, d, p) = self.eng.result();
        (c, d, p, out.converged, out.iterations, out.max_iter, out.weight)
    }
}

#[pymethods]
impl PyRelayEngine {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (num_checks, columns, priors, kind, frac_bits, relay, pre_iter, legs, leg_iter, solutions, stop_after_first, gamma0, gamma_range, gammas, alpha, alpha_scaling, data_scale, max_data, logging, seed, values=None))]
    fn new(
        num_checks: usize,
        columns: Vec<Vec<u32>>,
        priors: Vec<f64>,
        kind: &str,
        frac_bits: Option<i64>,
        relay: bool,
        pre_iter: usize,
        legs: usize,
        leg_iter: usize,
        solutions: Option<usize>,
        stop_after_first: bool,
        gamma0: Option<f64>,
        gamma_range: (f64, f64),
        gammas: Option<Vec<Vec<f64>>>,
        alpha: Option<f64>,
        alpha_scaling: f64,
        data_scale: Option<f64>,
        max_data: Option<f64>,
        logging: bool,
        seed: u64,
        values: Option<Vec<Vec<u8>>>,
    ) -> PyResult<Self> {
        use crate::relay_num::Fixed;
        let config = RelayConfig { pre_iter, legs, leg_iter, solutions, stop_after_first, gamma0, gamma_range, gammas, alpha, alpha_scaling, data_scale, max_data, logging: logging && relay, seed };
        let (n, c, v, p) = (num_checks, &columns, values.as_deref(), &priors);
        macro_rules! fixed {
            ($($f:literal)*) => {
                match frac_bits {
                    $(Some($f) => relay_engine::<Fixed<$f>>(n, c, v, p, config)?,)*
                    other => return Err(err(format!("Decoder does not support {other:?} fractional bits"))),
                }
            };
        }
        let eng = match kind {
            "f32" => relay_engine::<f32>(n, c, v, p, config)?,
            "f64" => relay_engine::<f64>(n, c, v, p, config)?,
            "i8" => relay_engine::<i8>(n, c, v, p, config)?,
            "i16" => relay_engine::<i16>(n, c, v, p, config)?,
            "i32" => relay_engine::<i32>(n, c, v, p, config)?,
            "i64" => relay_engine::<i64>(n, c, v, p, config)?,
            "fixed" => fixed!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31),
            _ => return Err(err(format!("unknown number type {kind:?}"))),
        };
        Ok(PyRelayEngine { eng, relay, num_checks, num_vars: columns.len() })
    }

    /// The decoder with its state as it is now, to go on independently.
    fn copy(&self) -> Self {
        PyRelayEngine { eng: self.eng.boxed_clone(), relay: self.relay, num_checks: self.num_checks, num_vars: self.num_vars }
    }

    fn decode(&mut self, syndrome: Vec<u8>) -> PyResult<RelayResult> {
        checked_syndrome(&syndrome, self.num_checks)?;
        Ok(self.one(&syndrome))
    }

    /// `count` rows of syndrome bytes (one per check), without the GIL: each in turn, the state
    /// going on from row to row; or with `fresh`, each from the state the decoder has now (as
    /// IBM's parallel batches decode, each row on a copy), leaving it unchanged, on `threads`
    /// threads (0: every core). Returns the corrections (one byte per fault per row), the
    /// checks they set off (one byte per check per row), the posteriors (little-endian f64 per
    /// fault per row, if asked for, else empty), each row's (converged, iterations, iteration
    /// limit, weight), and each row's log lines (if asked for, else none).
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    #[pyo3(signature = (rows, count, fresh, threads=0, posteriors=true, logs=false))]
    fn decode_rows<'py>(
        &mut self,
        py: Python<'py>,
        rows: &[u8],
        count: usize,
        fresh: bool,
        threads: usize,
        posteriors: bool,
        logs: bool,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, Vec<(bool, usize, usize, f64)>, Vec<Vec<String>>)> {
        check_rows(rows.len(), self.num_checks, count, self.num_checks, "syndrome bits")?;
        let relay = self.relay;
        let eng = &mut self.eng;
        let out = py.detach(move || eng.batch(rows, count, relay, fresh, threads, posteriors, logs));
        let corrections: Vec<u8> = out.iter().flat_map(|r| r.correction.iter().copied()).collect();
        let decoded: Vec<u8> = out.iter().flat_map(|r| r.decoded.iter().copied()).collect();
        let post = le_f64(out.iter().flat_map(|r| r.posteriors.iter().copied()));
        let stats = out.iter().map(|r| (r.outcome.converged, r.outcome.iterations, r.outcome.max_iter, r.outcome.weight)).collect();
        let lines = if logs { out.into_iter().map(|r| r.log).collect() } else { Vec::new() };
        Ok((PyBytes::new(py, &corrections), PyBytes::new(py, &decoded), PyBytes::new(py, &post), stats, lines))
    }

    /// IBM's decoding quality of an error: Σ ln((1 − p)/p) over its faults (finite ones).
    fn quality(&self, error: Vec<u8>) -> PyResult<f64> {
        if error.len() != self.num_vars {
            return Err(err(format!("{} error bits for {} faults", error.len(), self.num_vars)));
        }
        Ok(self.eng.quality(&error))
    }

    /// IBM's Relay log lines for the last decode.
    fn log_lines(&self) -> Vec<String> {
        self.eng.log_lines()
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
    RelayConfig { pre_iter, legs, leg_iter, solutions, gamma0, gamma_range, gammas, alpha, alpha_scaling, seed, ..RelayConfig::default() }
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

    /// One shot's lifting of a Möbius matching given as its edges, in a matcher's order (the
    /// boundary as -1): the observables, None where no lifting explains the shot.
    fn lift_edges(&self, defects: Vec<u32>, edges: Vec<(i64, i64)>) -> PyResult<Option<u64>> {
        if let Some(&d) = defects.iter().find(|&&d| d as usize >= self.num_detectors) {
            return Err(err(format!("detector {d} is beyond the model's {} detectors", self.num_detectors)));
        }
        let n = self.dec.num_mobius_nodes() as i64;
        let mut given = Vec::with_capacity(edges.len());
        for (a, b) in edges {
            let node = |x: i64| if x == -1 { Ok(n as u32) } else if (0..n).contains(&x) { Ok(x as u32) } else { Err(err(format!("Möbius node {x} is beyond the {n} nodes"))) };
            given.push((node(a)?, node(b)?));
        }
        let mut sorted = defects.clone();
        sorted.sort_unstable();
        sorted.dedup();
        let fired = |d: u32| sorted.binary_search(&d).is_ok();
        let mut work = self.dec.work();
        Ok(self.dec.decode_given_edges(&sorted, &given, &fired, &mut work).ok())
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
    #[pyo3(signature = (dem, beam, beam_climbing, no_revisit, queue_limit, orders, num_orders, method, seed, penalty, merge, flavour, sparsify=None, visualize=false, verbose=false))]
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
        sparsify: Option<(i64, i64, i64)>,
        visualize: bool,
        verbose: bool,
    ) -> PyResult<Self> {
        use crate::search::{Flavour, OrderMethod};
        let d = dem.flat()?;
        let flavour = match flavour {
            "native" => Flavour::native(),
            "libc++" => Flavour::LibCxx,
            "libstdc++" => Flavour::LibStdCxx,
            "msvc" => Flavour::Msvc,
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
        let config = crate::search::SearchConfig { beam, beam_climbing, no_revisit, queue_limit, orders: orders.clone(), detector_penalty: penalty, merge_errors: merge, flavour, sparsify: sparsify.map(|(base_degree, max_degree, reactivate_limit)| crate::search::Sparsify { base_degree, max_degree, reactivate_limit }), visualize, verbose };
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

    /// Change the settings a decode reads (Tesseract reads them from its decoder's config at
    /// every decode): the beam, beam climbing, revisiting, the queue bound, the detector
    /// orders and the detector penalty.
    #[allow(clippy::too_many_arguments)]
    fn reconfigure(&mut self, beam: Option<usize>, beam_climbing: bool, no_revisit: bool, queue_limit: Option<usize>, orders: Vec<Vec<u32>>, penalty: f64) -> PyResult<()> {
        self.dec.reconfigure(beam, beam_climbing, no_revisit, queue_limit, orders.clone(), penalty).map_err(err)?;
        self.orders = orders;
        Ok(())
    }

    /// The model faults one shot's search found (by index in the flattened model).
    fn decode_to_faults(&self, defects: Vec<u32>) -> (Vec<usize>, bool) {
        let found = self.dec.decode(&defects);
        (found.faults, found.low_confidence)
    }

    /// One search with detector order `order` and beam `beam`: (model faults, gave up).
    fn decode_once(&self, defects: Vec<u32>, order: usize, beam: usize) -> PyResult<(Vec<usize>, bool)> {
        if order >= self.orders.len() {
            return Err(err(format!("det_order {order} is beyond the {} detector orders", self.orders.len())));
        }
        if let Some(&d) = defects.iter().find(|&&d| d as usize >= self.num_detectors) {
            return Err(err(format!("Symptom {d} references a detector >= num_detectors (= {}).", self.num_detectors)));
        }
        let found = self.dec.decode_once(&defects, order, beam);
        Ok((found.faults, found.low_confidence))
    }

    /// The faults searched over, after merging and dropping those of probability 0: (cost,
    /// detectors, observables).
    fn retained(&self) -> Vec<(f64, Vec<u32>, Vec<u32>)> {
        self.dec.retained().map(|(c, d, o)| (c, d.to_vec(), (0..64).filter(|k| o >> k & 1 == 1).collect())).collect()
    }

    /// The visualizer's lines recorded since the last call.
    fn take_trace(&self) -> Vec<String> {
        self.dec.take_trace()
    }

    /// The verbose log's lines since the last call.
    fn take_log(&self) -> Vec<String> {
        self.dec.take_log()
    }

    /// One shot's search with some model faults' probabilities changed for it alone:
    /// (model faults, gave up, their cost under the changed probabilities).
    fn decode_with_probabilities(&mut self, defects: Vec<u32>, updates: Vec<(usize, f64)>) -> PyResult<(Vec<usize>, bool, f64)> {
        if let Some(&d) = defects.iter().find(|&&d| d as usize >= self.num_detectors) {
            return Err(err(format!("Symptom {d} references a detector >= num_detectors (= {}).", self.num_detectors)));
        }
        let (found, cost) = self.dec.decode_with_probabilities(&defects, &updates).map_err(err)?;
        Ok((found.faults, found.low_confidence, cost))
    }

    /// The retained fault each model fault went into (-1 for one dropped).
    fn model_to_fault(&self) -> Vec<i64> {
        self.dec.model_to_fault().iter().map(|&f| if f == usize::MAX { -1 } else { f as i64 }).collect()
    }
}

/// CUDA-Q QEC's lookup table (`single_error_lut`, `multi_error_lut`) over a check matrix.
#[pyclass(name = "LookupTable", module = "stabilizer_qec._core", frozen)]
pub struct PyLookupTable {
    table: crate::lookup::LookupTable,
}

#[pymethods]
impl PyLookupTable {
    #[new]
    fn new(num_rows: usize, columns: Vec<Vec<u32>>, depth: usize, rates: Vec<f64>) -> PyResult<Self> {
        Ok(PyLookupTable { table: crate::lookup::LookupTable::new(num_rows, &columns, depth, &rates).map_err(err)? })
    }

    /// The faults entered for the rows that fired, and their probability; None if none are.
    fn lookup(&self, fired: Vec<u32>) -> Option<(Vec<u32>, f64)> {
        self.table.lookup(&fired).map(|(e, p)| (e.to_vec(), p))
    }

    fn __len__(&self) -> usize {
        self.table.len()
    }
}

/// CUDA-Q QEC's `generate_random_pcm`: the (row, column) of each 1, drawn from
/// `std::mt19937_64(seed)` through libstdc++'s distributions (CUDA-Q QEC ships only Linux builds).
#[pyfunction]
fn cudaq_random_pcm(n_rounds: usize, n_errs_per_round: usize, n_syndromes_per_round: usize, weight: i64, seed: u64) -> PyResult<Vec<(usize, usize)>> {
    use crate::cpp_random::{Library, Mt19937_64};
    if weight < 0 || weight as usize > n_syndromes_per_round {
        return Err(err(format!("generate_random_pcm: weight must be in [0, n_syndromes_per_round]; got weight={weight}, n_syndromes_per_round={n_syndromes_per_round}")));
    }
    let lib = Library::LibStdCxx;
    let mut g = Mt19937_64::new(seed);
    let mut out = Vec::new();
    for r in 0..n_rounds {
        for c in 0..n_errs_per_round {
            let c_ix = r * n_errs_per_round + c;
            let all_in_round = lib.uniform(&mut g, 0, 1) == 1 || r == n_rounds - 1;
            let row_max = if all_in_round { n_syndromes_per_round } else { 2 * n_syndromes_per_round };
            if row_max == 0 {
                continue;
            }
            let mut taken = vec![false; row_max];
            for _ in 0..weight {
                let mut row_ix = lib.uniform(&mut g, 0, row_max as u64 - 1) as usize;
                while taken[row_ix] {
                    row_ix = lib.uniform(&mut g, 0, row_max as u64 - 1) as usize;
                }
                taken[row_ix] = true;
                out.push((r * n_syndromes_per_round + row_ix, c_ix));
            }
        }
    }
    Ok(out)
}

/// Exact maximum-likelihood masses (mass_0, mass_1) of a logical given a soft syndrome
/// (`tn_ml::masses`), for each syndrome.
#[pyfunction]
fn tn_ml_masses(num_checks: usize, columns: Vec<Vec<u32>>, logical: Vec<bool>, priors: Vec<f64>, syndromes: Vec<Vec<f64>>, order: Vec<usize>) -> PyResult<Vec<(f64, f64)>> {
    syndromes.iter().map(|s| crate::tn_ml::masses(num_checks, &columns, &logical, &priors, s, &order).map_err(err)).collect()
}

/// Minimum-weight perfect matching on a graph of `n` nodes given by its edges (u, v, integer
/// weight), as Blossom V's `mwpm` (qecsim's interface): each node's mate.
#[pyfunction]
fn blossom_mwpm_ids(n: usize, edges: Vec<(usize, usize, i64)>) -> PyResult<Vec<usize>> {
    crate::blossom::mwpm_on_edges(n, &edges).map_err(err)
}

/// `std::shuffle` of 0..n by `std::mt19937_64(seed)`, as libstdc++ does it (CUDA-Q QEC's
/// `shuffle_pcm_columns`; CUDA-Q QEC ships only Linux builds).
#[pyfunction]
fn cpp_shuffled_order(n: usize, seed: u64) -> Vec<u32> {
    use crate::cpp_random::{Library, Mt19937_64};
    let mut v: Vec<u32> = (0..n as u32).collect();
    Library::LibStdCxx.shuffle(&mut v, &mut Mt19937_64::new(seed));
    v
}

/// Tesseract's generated detector orders for a model (`build_det_orders`).
#[pyfunction]
#[pyo3(signature = (dem, count, method, seed, flavour))]
fn search_orders(dem: &PyDem, count: usize, method: &str, seed: u64, flavour: &str) -> PyResult<Vec<Vec<u32>>> {
    use crate::search::{Flavour, OrderMethod};
    let d = dem.flat()?;
    let flavour = match flavour {
        "native" => Flavour::native(),
        "libc++" => Flavour::LibCxx,
        "libstdc++" => Flavour::LibStdCxx,
        "msvc" => Flavour::Msvc,
        other => return Err(err(format!("unknown flavour '{other}'"))),
    };
    let method = match method {
        "index" => OrderMethod::Index,
        "bfs" => OrderMethod::Bfs,
        "coordinate" => OrderMethod::Coordinate,
        other => return Err(err(format!("unknown detector order method '{other}'"))),
    };
    Ok(crate::search::generated_orders(d, count, method, seed, flavour))
}

/// The cheapest set of faults (by index) setting off exactly `detections`, found by an exact
/// search (no beam, no pruning): (faults, whether none exists).
#[pyfunction]
fn search_exact(num_detectors: usize, costs: Vec<f64>, detectors: Vec<Vec<u32>>, detections: Vec<u32>) -> PyResult<(Vec<usize>, bool)> {
    if costs.len() != detectors.len() {
        return Err(err("one cost per fault".into()));
    }
    let faults = costs.into_iter().zip(detectors).map(|(c, d)| (c, d, 0u64)).collect();
    let config = crate::search::SearchConfig { beam: Some(usize::MAX / 4), beam_climbing: false, no_revisit: false, queue_limit: None, orders: vec![(0..num_detectors as u32).collect()], detector_penalty: 0.0, merge_errors: false, flavour: crate::search::Flavour::native(), sparsify: None, visualize: false, verbose: false };
    let s = crate::search::Search::from_faults(num_detectors, faults, config).map_err(err)?;
    let found = s.decode(&detections);
    Ok((found.faults, found.low_confidence))
}

/// Tesseract's `build_detector_graph`: each detector's neighbours, sorted.
#[pyfunction]
fn search_detector_graph(dem: &PyDem) -> PyResult<Vec<Vec<u32>>> {
    Ok(crate::search::detector_graph(dem.flat()?))
}

/// Tesseract's `suggest_sparsify_reactivate_limit`.
#[pyfunction]
fn suggest_sparsify_reactivate_limit(num_detectors: usize, sparsify_base_degree: i64) -> PyResult<i64> {
    crate::search::suggest_reactivate_limit(num_detectors, sparsify_base_degree, i32::MAX as i64).map_err(err)
}

/// ρ's entries as little-endian complex128 bytes.
fn density_bytes<'py>(py: Python<'py>, rho: &crate::density::DensityMatrix) -> Bound<'py, PyBytes> {
    let m = rho.matrix();
    let mut out = Vec::with_capacity(m.len() * 16);
    for c in m {
        out.extend_from_slice(&c.re.to_le_bytes());
        out.extend_from_slice(&c.im.to_le_bytes());
    }
    PyBytes::new(py, &out)
}

/// The density-matrix simulator behind `DensityMatrixSimulator`.
#[pyclass(name = "DensityMatrixCore", module = "stabilizer_qec._core")]
pub struct PyDensitySim {
    rho: crate::density::DensityMatrix,
    rec: Vec<bool>,
    rng: crate::surface_code::Xorshift,
}

#[pymethods]
impl PyDensitySim {
    #[new]
    fn new(n: usize, seed: u64) -> PyResult<Self> {
        if n > crate::density::MAX_QUBITS {
            return Err(err(format!("at most {} qubits", crate::density::MAX_QUBITS)));
        }
        // The seed mixed, as every sampler's: Xorshift's first draws from small seeds are tiny.
        Ok(PyDensitySim { rho: crate::density::DensityMatrix::new(n), rec: Vec::new(), rng: crate::surface_code::Xorshift::new(crate::batch_sampler::batch_seed(seed, 0)) })
    }

    fn copy(&self, seed: u64) -> Self {
        PyDensitySim { rho: self.rho.clone(), rec: self.rec.clone(), rng: crate::surface_code::Xorshift::new(crate::batch_sampler::batch_seed(seed, 0)) }
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.rho.num_qubits()
    }

    fn grow(&mut self, n: usize) -> PyResult<()> {
        if n > crate::density::MAX_QUBITS {
            return Err(err(format!("at most {} qubits", crate::density::MAX_QUBITS)));
        }
        self.rho.grow(n);
        Ok(())
    }

    // This and `apply_channel` run holding the interpreter's lock: releasing it while the
    // simulator is borrowed would let another thread's call on it fail, not wait.
    fn do_circuit(&mut self, circuit: &PyCircuit) -> PyResult<()> {
        // Lookbacks may read the records of earlier circuits, as TableauSimulator's do: as many
        // of them as the deepest reaches go first, as pads, and are dropped after.
        let k = (deepest_lookback(&circuit.circuit.inner.instrs) as usize).min(self.rec.len());
        let padded;
        let inner = if k > 0 {
            let pads: Vec<&str> = self.rec[self.rec.len() - k..].iter().map(|&b| if b { "1" } else { "0" }).collect();
            padded = crate::circuit::Circuit::parse(&format!("MPAD {}\n{}", pads.join(" "), circuit.circuit.inner.to_stim())).map_err(err)?;
            &padded
        } else {
            &circuit.circuit.inner
        };
        let program = crate::density::DensityProgram::with_layout(inner, true, crate::statevec::Outcomes::StateOnly).map_err(err)?;
        let (rho, rec) = program.sample(self.rho.clone(), &mut self.rng).map_err(err)?;
        self.rho = rho;
        self.rec.extend(&rec[k..]);
        Ok(())
    }

    fn record(&self) -> Vec<bool> {
        self.rec.clone()
    }

    fn matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        density_bytes(py, &self.rho)
    }

    fn set_matrix(&mut self, n: usize, data: &[u8]) -> PyResult<()> {
        self.rho = crate::density::DensityMatrix::from_matrix(n, &complex_entries(data)).map_err(err)?;
        Ok(())
    }

    fn expectation(&self, paulis: Vec<(usize, u8)>) -> (f64, f64) {
        let e = self.rho.expectation(&paulis);
        (e.re, e.im)
    }

    fn purity(&self) -> f64 {
        self.rho.purity()
    }

    fn apply_matrix(&mut self, qubits: Vec<usize>, data: &[u8]) -> PyResult<()> {
        self.rho.apply_matrix(&qubits, &complex_entries(data)).map_err(err)
    }

    fn apply_kraus(&mut self, qubits: Vec<usize>, ops: Vec<Vec<u8>>) -> PyResult<()> {
        let ops: Vec<Vec<crate::statevec::C>> = ops.iter().map(|d| complex_entries(d)).collect();
        self.rho.apply_kraus(&qubits, &ops).map_err(err)
    }

    #[pyo3(signature = (qubit, forced=None))]
    fn measure(&mut self, qubit: usize, forced: Option<bool>) -> PyResult<bool> {
        if qubit >= self.rho.num_qubits() {
            return Err(err(format!("qubit {qubit} is not below {}", self.rho.num_qubits())));
        }
        let outcome = self.rho.measure(qubit, forced, &mut self.rng);
        self.rec.push(outcome);
        Ok(outcome)
    }

    /// The circuit as a channel on the current state: records averaged away, none kept.
    fn apply_channel(&mut self, circuit: &PyCircuit, max_branches: usize) -> PyResult<()> {
        let program = crate::density::DensityProgram::with_layout(&circuit.circuit.inner, true, crate::statevec::Outcomes::StateOnly).map_err(err)?;
        self.rho = program.final_state(Some(self.rho.clone()), max_branches).map_err(err)?;
        Ok(())
    }
}

/// The deepest record a circuit's instructions read back (`k` of `rec[-k]`), loops included.
fn deepest_lookback(instrs: &[crate::circuit::Instr]) -> u32 {
    use crate::circuit::{Control, Instr};
    instrs
        .iter()
        .map(|i| match i {
            Instr::Detector { recs, .. } | Instr::Observable { recs, .. } => recs.iter().copied().max().unwrap_or(0),
            Instr::Feedback { control: Control::Rec(k), .. } => *k,
            Instr::Repeat { body, .. } | Instr::Gate { body, .. } => deepest_lookback(body),
            _ => 0,
        })
        .max()
        .unwrap_or(0)
}

/// Little-endian complex128 bytes as numbers.
fn complex_entries(data: &[u8]) -> Vec<crate::statevec::C> {
    data.chunks_exact(16).map(|c| crate::statevec::C::new(f64::from_le_bytes(c[..8].try_into().unwrap()), f64::from_le_bytes(c[8..].try_into().unwrap()))).collect()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDensitySim>()?;
    m.add_class::<PyCircuit>()?;
    m.add_class::<PyDem>()?;
    m.add_class::<PySampler>()?;
    m.add_class::<PyDemSampler>()?;
    m.add_class::<PyMeasurementSampler>()?;
    m.add_class::<PyM2d>()?;
    m.add_class::<PyMatcher>()?;
    m.add_class::<PyUnionFinder>()?;
    m.add_class::<PyDemCollisionClustering>()?;
    m.add_class::<PyDemLocalClustering>()?;
    m.add_class::<PyDemArgumentReweighting>()?;
    m.add_class::<PyDemLeakageAware>()?;
    m.add_class::<PyPreDecoder>()?;
    m.add_class::<PyPecosSparseSim>()?;
    m.add_class::<PyDemAmbiguityClustering>()?;
    m.add_class::<PyBeliefMatcher>()?;
    m.add_class::<PyDemBpOsd>()?;
    m.add_class::<PyWindowMatcher>()?;
    m.add_class::<PyBp>()?;
    m.add_class::<PyBpOsd>()?;
    m.add_class::<PyDemBpLsd>()?;
    m.add_class::<PyBpLsd>()?;
    m.add_class::<PyFlip>()?;
    m.add_class::<PyLdpcUf>()?;
    m.add_class::<PyMbp>()?;
    m.add_class::<PyRelayEngine>()?;
    m.add_class::<PyRelay>()?;
    m.add_class::<PyDemRelay>()?;
    m.add_class::<PyDemColorMatcher>()?;
    m.add_class::<PyDemSearch>()?;
    m.add_class::<PyLookupTable>()?;
    m.add_function(wrap_pyfunction!(cudaq_random_pcm, m)?)?;
    m.add_function(wrap_pyfunction!(cpp_shuffled_order, m)?)?;
    m.add_function(wrap_pyfunction!(tn_ml_masses, m)?)?;
    m.add_function(wrap_pyfunction!(blossom_mwpm_ids, m)?)?;
    m.add_function(wrap_pyfunction!(search_orders, m)?)?;
    m.add_function(wrap_pyfunction!(search_detector_graph, m)?)?;
    m.add_function(wrap_pyfunction!(search_exact, m)?)?;
    m.add_function(wrap_pyfunction!(suggest_sparsify_reactivate_limit, m)?)?;
    Ok(())
}

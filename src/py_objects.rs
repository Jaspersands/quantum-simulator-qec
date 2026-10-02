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
use crate::batch::{belief_shots, bposd_shots, match_shots, window_info, window_shots, WindowInfo};
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

fn api_err(e: crate::api::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
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

    fn repeated(&self, count: u64) -> PyResult<PyCircuit> {
        Ok(PyCircuit { circuit: self.circuit.repeated(count).map_err(api_err)? })
    }

    fn copy(&self) -> PyCircuit {
        PyCircuit { circuit: self.circuit.clone() }
    }

    /// The detector error model; `decompose` splits faults into graph-like pieces as Stim does,
    /// `approximate` is Stim's `approximate_disjoint_errors` as a threshold (None: off), and
    /// `flatten` writes it without folding its loops.
    #[pyo3(signature = (decompose, approximate=None, flatten=false))]
    fn detector_error_model(&self, decompose: bool, approximate: Option<f64>, flatten: bool) -> PyResult<PyDem> {
        let options = crate::api::DemOptions::new().decompose_errors(decompose).approximate_disjoint_errors(approximate).flatten_loops(flatten);
        Ok(PyDem { dem: self.circuit.detector_error_model(&options).map_err(api_err)? })
    }

    fn sampler(&self, seed: u64) -> PyResult<PySampler> {
        Ok(PySampler { sampler: BatchSampler::new(&self.circuit.inner).map_err(err)?, seed, next: 0.into() })
    }

    fn m2d(&self) -> PyResult<PyM2d> {
        Ok(PyM2d { m2d: M2d::new(&self.circuit.inner).map_err(err)? })
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

    fn flattened(&self) -> PyResult<PyDem> {
        Ok(PyDem { dem: self.dem.flattened().map_err(api_err)? })
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

/// Window decoding of a model cut by time.
#[pyclass(name = "WindowMatcher", module = "stabilizer_qec._core")]
pub struct PyWindowMatcher {
    wd: WindowDecoder,
    correlated: bool,
    #[pyo3(get)]
    num_detectors: usize,
    #[pyo3(get)]
    num_observables: usize,
}

#[pymethods]
impl PyWindowMatcher {
    #[new]
    fn new(dem: &PyDem, commit: usize, buffer: usize, mode: &str, correlated: bool) -> PyResult<Self> {
        let mode = match mode {
            "sliding" => Mode::Sliding,
            "parallel" => Mode::Parallel,
            other => return Err(err(format!("mode '{other}' is neither sliding nor parallel"))),
        };
        let d = dem.flat()?;
        let wd = WindowDecoder::new(Model::new(d).map_err(err)?, commit, buffer, mode).map_err(err)?;
        Ok(PyWindowMatcher { wd, correlated, num_detectors: d.num_detectors, num_observables: d.num_observables })
    }

    fn windows(&self) -> Vec<WindowInfo> {
        window_info(&self.wd)
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
        let out = py.detach(|| window_shots(wd, packed, nd, shots, correlated, threads, timings));
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCircuit>()?;
    m.add_class::<PyDem>()?;
    m.add_class::<PySampler>()?;
    m.add_class::<PyM2d>()?;
    m.add_class::<PyMatcher>()?;
    m.add_class::<PyBeliefMatcher>()?;
    m.add_class::<PyDemBpOsd>()?;
    m.add_class::<PyWindowMatcher>()?;
    m.add_class::<PyBp>()?;
    m.add_class::<PyBpOsd>()?;
    Ok(())
}

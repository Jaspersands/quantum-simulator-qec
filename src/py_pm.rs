//! Python bindings for PyMatching's graph API (`user_graph`): the graph, built and queried, and
//! its decoder (`pm_exact`, PyMatching's sparse blossom), rebuilt on first use after any change.
//! Single shots decode on one kept `Mwpm`, as PyMatching keeps one; `decode_batch` gives each
//! thread its own.

use std::sync::Mutex;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::pm_exact::Mwpm;
use crate::user_graph::{Merge, UserDecoder, UserGraph};

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

type EdgeRow = (usize, Option<usize>, Vec<usize>, f64, f64);

/// The sparse blossom's recursions (shattering nested blossoms) get room on decoding threads.
const DECODE_THREAD_STACK: usize = 64 << 20;

struct Engine {
    decoder: UserDecoder,
    mwpm: Mwpm,
}

#[pyclass(name = "UserGraph", module = "stabilizer_qec._core")]
pub struct PyUserGraph {
    g: UserGraph,
    engine: Mutex<Option<Engine>>,
}

impl PyUserGraph {
    fn changed(&mut self) {
        *self.engine.get_mut().expect("not poisoned") = None;
    }

    /// Runs `f` on the kept decoder (built first if the graph changed), without the GIL.
    fn with_engine<R: Send>(&self, py: Python<'_>, f: impl FnOnce(&mut Engine) -> Result<R, String> + Send) -> PyResult<R> {
        let g = &self.g;
        let engine = &self.engine;
        py.detach(|| {
            let mut slot = engine.lock().map_err(|_| "the decoder's lock was poisoned".to_string())?;
            if slot.is_none() {
                let decoder = g.decoder()?;
                let mwpm = decoder.mwpm();
                *slot = Some(Engine { decoder, mwpm });
            }
            f(slot.as_mut().expect("just built"))
        })
        .map_err(err)
    }
}

#[pymethods]
impl PyUserGraph {
    #[new]
    #[pyo3(signature = (num_nodes=0, num_fault_ids=0))]
    fn new(num_nodes: usize, num_fault_ids: usize) -> Self {
        PyUserGraph { g: UserGraph::new(num_nodes, num_fault_ids), engine: Mutex::new(None) }
    }

    /// PyMatching's graph of a model (see `UserGraph::from_dem`).
    #[staticmethod]
    fn from_dem(text: &str) -> PyResult<Self> {
        let dem = crate::dem_program::DemProgram::parse(text).map_err(err)?;
        Ok(PyUserGraph { g: UserGraph::from_dem(&dem).map_err(err)?, engine: Mutex::new(None) })
    }

    fn copy(&self) -> Self {
        PyUserGraph { g: self.g.clone(), engine: Mutex::new(None) }
    }

    fn add_edge(&mut self, node1: usize, node2: usize, fault_ids: Vec<usize>, weight: f64, error_probability: f64, merge_strategy: &str) -> PyResult<()> {
        let how = Merge::parse(merge_strategy).map_err(err)?;
        self.g.add_edge(node1, node2, &fault_ids, weight, error_probability, how).map_err(err)?;
        self.changed();
        Ok(())
    }

    fn add_boundary_edge(&mut self, node: usize, fault_ids: Vec<usize>, weight: f64, error_probability: f64, merge_strategy: &str) -> PyResult<()> {
        let how = Merge::parse(merge_strategy).map_err(err)?;
        self.g.add_boundary_edge(node, &fault_ids, weight, error_probability, how).map_err(err)?;
        self.changed();
        Ok(())
    }

    fn set_boundary(&mut self, nodes: Vec<usize>) {
        self.g.set_boundary(nodes.into_iter().collect());
        self.changed();
    }

    fn ensure_num_fault_ids(&mut self, n: usize) {
        self.g.ensure_num_fault_ids(n);
        self.changed();
    }

    fn boundary(&self) -> Vec<usize> {
        self.g.boundary().iter().copied().collect()
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.g.num_nodes()
    }

    #[getter]
    fn num_edges(&self) -> usize {
        self.g.num_edges()
    }

    #[getter]
    fn num_fault_ids(&self) -> usize {
        self.g.num_fault_ids()
    }

    #[getter]
    fn num_detectors(&self) -> usize {
        self.g.num_detectors()
    }

    fn all_edges_have_error_probabilities(&self) -> bool {
        self.g.all_edges_have_error_probabilities()
    }

    /// The edge between `node` and `other` (None: the boundary): (fault ids, weight,
    /// probability), or None.
    #[pyo3(signature = (node, other=None))]
    fn edge_data(&self, node: usize, other: Option<usize>) -> Option<(Vec<usize>, f64, f64)> {
        let e = &self.g.edges[self.g.find(node, other)?];
        Some((e.fault_ids.clone(), e.weight, e.error_probability))
    }

    /// Every edge, in the order added: (node1, node2 or None, fault ids, weight, probability).
    fn edges(&self) -> Vec<EdgeRow> {
        self.g.edges.iter().map(|e| (e.node1, e.node2, e.fault_ids.clone(), e.weight, e.error_probability)).collect()
    }

    /// One shot (detection events as node indices): (flipped fault ids as one byte each,
    /// weight).
    fn decode<'py>(&self, py: Python<'py>, events: Vec<u64>) -> PyResult<(Bound<'py, PyBytes>, f64)> {
        let (bytes, weight) = self.with_engine(py, |e| e.mwpm.try_decode(&events))?;
        Ok((PyBytes::new(py, &bytes), weight))
    }

    /// Shots as bit-packed rows of `row_bytes` bytes, bit k a detection event on node k:
    /// (predictions bit-packed per shot, `ceil(num_fault_ids / 8)` bytes each; weights as f64).
    /// Shots are split into contiguous runs, one per thread; a failure is the earliest shot's,
    /// PyMatching's message after `shot k: `.
    fn decode_batch<'py>(&self, py: Python<'py>, packed: &[u8], shots: usize, row_bytes: usize, threads: usize) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        if packed.len() != shots * row_bytes {
            return Err(err(format!("{} bytes of shots, not {shots} rows of {row_bytes}", packed.len())));
        }
        let (preds, weights) = self.with_engine(py, |e| {
            let nf = e.decoder.num_fault_ids();
            let out_bytes = nf.div_ceil(8);
            // At least a byte a shot, so a graph with no fault ids still decodes every shot.
            let stride = out_bytes.max(1);
            let mut preds = vec![0u8; shots * stride];
            let mut weights = vec![0f64; shots];
            let threads = if threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { threads }.max(1).min(shots.max(1));
            let chunk = shots.div_ceil(threads).max(1);
            let decoder = &e.decoder;
            let mut kept = Some(&mut e.mwpm);
            let results: Vec<Result<(), String>> = std::thread::scope(|scope| {
                let handles: Vec<_> = preds
                    .chunks_mut(chunk * stride)
                    .zip(weights.chunks_mut(chunk))
                    .enumerate()
                    .map(|(c, (pout, wout))| {
                        let kept = kept.take();
                        std::thread::Builder::new()
                            .stack_size(DECODE_THREAD_STACK)
                            .spawn_scoped(scope, move || -> Result<(), String> {
                                let mut own;
                                let mwpm = match kept {
                                    Some(m) => m,
                                    None => {
                                        own = decoder.mwpm();
                                        &mut own
                                    }
                                };
                                let mut events: Vec<u64> = Vec::new();
                                for (i, w) in wout.iter_mut().enumerate() {
                                    let shot = c * chunk + i;
                                    let row = &packed[shot * row_bytes..(shot + 1) * row_bytes];
                                    events.clear();
                                    for (b, &byte) in row.iter().enumerate() {
                                        let mut bits = byte;
                                        while bits != 0 {
                                            events.push((b * 8 + bits.trailing_zeros() as usize) as u64);
                                            bits &= bits - 1;
                                        }
                                    }
                                    let o = &mut pout[i * stride..i * stride + out_bytes];
                                    *w = decoder.decode_packed(mwpm, &events, o).map_err(|e| format!("shot {shot}: {e}"))?;
                                }
                                Ok(())
                            })
                            .expect("a decoding thread starts")
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().expect("a decoding thread panicked")).collect()
            });
            if let Some(Err(e)) = results.into_iter().find(|r| r.is_err()) {
                return Err(e);
            }
            if out_bytes == 0 {
                preds.clear();
            }
            Ok((preds, weights))
        })?;
        let weights: Vec<u8> = weights.iter().flat_map(|w| w.to_le_bytes()).collect();
        Ok((PyBytes::new(py, &preds), PyBytes::new(py, &weights)))
    }

    /// The correction's edges, as node pairs (-1: the boundary), in PyMatching's order.
    fn decode_to_edges(&self, py: Python<'_>, events: Vec<u64>) -> PyResult<Vec<(i64, i64)>> {
        self.with_engine(py, |e| e.mwpm.try_decode_to_edges(&events))
    }

    /// The matched pairs of detection events (-1: the boundary), in PyMatching's order.
    fn decode_to_matched_pairs(&self, py: Python<'_>, events: Vec<u64>) -> PyResult<Vec<(i64, i64)>> {
        self.with_engine(py, |e| e.mwpm.try_decode_to_match_edges(&events))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyUserGraph>()?;
    Ok(())
}

//! Python bindings for PyMatching's graph API (`user_graph`): the graph, built and queried, and
//! its decoder, rebuilt on first use after any change.

use std::sync::Mutex;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::user_graph::{Merge, UserDecoder, UserGraph};

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

type EdgeRow = (usize, Option<usize>, Vec<usize>, f64, f64);

#[pyclass(name = "UserGraph", module = "stabilizer_qec._core")]
pub struct PyUserGraph {
    g: UserGraph,
    decoder: Mutex<Option<std::sync::Arc<UserDecoder>>>,
}

impl PyUserGraph {
    fn changed(&mut self) {
        *self.decoder.get_mut().expect("not poisoned") = None;
    }

    fn decoder(&self) -> PyResult<std::sync::Arc<UserDecoder>> {
        let mut slot = self.decoder.lock().map_err(|_| err("the decoder's lock was poisoned".into()))?;
        if slot.is_none() {
            *slot = Some(std::sync::Arc::new(self.g.decoder().map_err(err)?));
        }
        Ok(slot.as_ref().expect("just built").clone())
    }
}

#[pymethods]
impl PyUserGraph {
    #[new]
    #[pyo3(signature = (num_nodes=0, num_fault_ids=0))]
    fn new(num_nodes: usize, num_fault_ids: usize) -> Self {
        PyUserGraph { g: UserGraph::new(num_nodes, num_fault_ids), decoder: Mutex::new(None) }
    }

    /// PyMatching's graph of a model (see `UserGraph::from_dem`).
    #[staticmethod]
    fn from_dem(text: &str) -> PyResult<Self> {
        let dem = crate::dem_program::DemProgram::parse(text).map_err(err)?;
        Ok(PyUserGraph { g: UserGraph::from_dem(&dem).map_err(err)?, decoder: Mutex::new(None) })
    }

    fn copy(&self) -> Self {
        PyUserGraph { g: self.g.clone(), decoder: Mutex::new(None) }
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
        let d = self.decoder()?;
        let mut s = d.scratch();
        let r = py.detach(|| d.decode(&mut s, &events)).map_err(err)?;
        let bytes: Vec<u8> = r.faults.iter().map(|&b| b as u8).collect();
        Ok((PyBytes::new(py, &bytes), r.weight))
    }

    /// Shots as bit-packed rows of `row_bytes` bytes, bit k a detection event on node k:
    /// (predictions bit-packed per shot, `ceil(num_fault_ids / 8)` bytes each; weights as f64).
    fn decode_batch<'py>(&self, py: Python<'py>, packed: &[u8], shots: usize, row_bytes: usize, threads: usize) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        if packed.len() != shots * row_bytes {
            return Err(err(format!("{} bytes of shots, not {shots} rows of {row_bytes}", packed.len())));
        }
        let d = self.decoder()?;
        let nf = d.num_fault_ids();
        let out_bytes = nf.div_ceil(8);
        // At least a byte a shot, so a graph with no fault ids still decodes every shot.
        let stride = out_bytes.max(1);
        let mut preds = vec![0u8; shots * stride];
        let mut weights = vec![0f64; shots];
        let failure: Option<String> = py.detach(|| {
            let threads = if threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { threads }.max(1).min(shots.max(1));
            let chunk = shots.div_ceil(threads).max(1);
            let mut failure = None;
            std::thread::scope(|scope| {
                let handles: Vec<_> = preds
                    .chunks_mut(chunk * stride)
                    .zip(weights.chunks_mut(chunk))
                    .enumerate()
                    .map(|(c, (pout, wout))| {
                        let d = &d;
                        scope.spawn(move || -> Result<(), String> {
                            let mut s = d.scratch();
                            let (mut events, mut buf) = (Vec::new(), Vec::new());
                            for (i, w) in wout.iter_mut().enumerate() {
                                let shot = c * chunk + i;
                                let row = &packed[shot * row_bytes..(shot + 1) * row_bytes];
                                events.clear();
                                for (b, &byte) in row.iter().enumerate() {
                                    let mut bits = byte;
                                    while bits != 0 {
                                        events.push((b * 8 + bits.trailing_zeros() as usize) as u32);
                                        bits &= bits - 1;
                                    }
                                }
                                let o = &mut pout[i * stride..i * stride + out_bytes];
                                *w = d.decode_packed(&mut s, &events, &mut buf, o).map_err(|e| format!("shot {shot}: {e}"))?;
                            }
                            Ok(())
                        })
                    })
                    .collect();
                for h in handles {
                    if let Err(e) = h.join().expect("a decoding thread panicked") {
                        failure.get_or_insert(e);
                    }
                }
            });
            failure
        });
        if let Some(e) = failure {
            return Err(err(e));
        }
        let weights: Vec<u8> = weights.iter().flat_map(|w| w.to_le_bytes()).collect();
        if out_bytes == 0 {
            preds.clear();
        }
        Ok((PyBytes::new(py, &preds), PyBytes::new(py, &weights)))
    }

    /// The correction's edges, as node pairs (-1: the boundary).
    fn decode_to_edges(&self, py: Python<'_>, events: Vec<u64>) -> PyResult<Vec<(i64, i64)>> {
        let d = self.decoder()?;
        let mut s = d.scratch();
        py.detach(|| d.decode_to_edges(&mut s, &events)).map_err(err)
    }

    /// The matched pairs of detection events (-1: the boundary).
    fn decode_to_matched_pairs(&self, py: Python<'_>, events: Vec<u64>) -> PyResult<Vec<(i64, i64)>> {
        let d = self.decoder()?;
        let mut s = d.scratch();
        py.detach(|| d.decode_to_matched_pairs(&mut s, &events)).map_err(err)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyUserGraph>()?;
    Ok(())
}

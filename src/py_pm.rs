//! Python bindings for PyMatching's graph API (`user_graph`): the graph, built and queried, and
//! its decoder (`pm_exact`, PyMatching's sparse blossom), rebuilt on first use after any change.
//! Single shots decode on one kept `Mwpm`, as PyMatching keeps one; `decode_batch` gives each
//! thread its own.

use std::sync::Mutex;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::pm_exact::Mwpm;
use crate::user_graph::{ImpliedWeight, Merge, UserDecoder, UserGraph};

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

type EdgeRow = (usize, Option<usize>, Vec<usize>, f64, f64);

/// The sparse blossom's recursions (shattering nested blossoms) get room on decoding threads.
const DECODE_THREAD_STACK: usize = 64 << 20;

struct Engine {
    decoder: UserDecoder,
    mwpm: Mwpm,
    /// Whether PyMatching's mwpm would carry its search graph: built by
    /// `get_mwpm_with_search_graph`, or over 64 fault ids (`to_mwpm`).
    search: bool,
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

    /// Runs `f` on the kept decoder, without the GIL: PyMatching's `get_mwpm`, or with `search`
    /// its `get_mwpm_with_search_graph`. The first builds only after a change to the graph; the
    /// second also rebuilds an mwpm built without the search graph (unless the graph has no
    /// nodes, where both graphs have 0). A rebuild sees the fault-id count as it is now, which a
    /// kept mwpm does not after `ensure_num_fault_ids`. A failed build keeps the old one, as
    /// PyMatching's `_mwpm = to_mwpm(...)` does when `to_mwpm` throws.
    fn with_engine<R: Send>(&self, py: Python<'_>, search: bool, f: impl FnOnce(&mut Engine) -> Result<R, String> + Send) -> PyResult<R> {
        let g = &self.g;
        let engine = &self.engine;
        py.detach(|| {
            let mut slot = engine.lock().map_err(|_| "the decoder's lock was poisoned".to_string())?;
            let rebuild = match slot.as_ref() {
                None => true,
                Some(e) => search && !e.search && e.decoder.num_nodes() != 0,
            };
            if rebuild {
                let decoder = g.decoder()?;
                let mwpm = decoder.mwpm();
                let search = search || decoder.num_fault_ids() > 64;
                *slot = Some(Engine { decoder, mwpm, search });
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

    /// PyMatching's graph of a model (see `UserGraph::from_dem`), or with `enable_correlations`
    /// its correlated load (`UserGraph::from_dem_with_correlations`).
    #[staticmethod]
    #[pyo3(signature = (text, enable_correlations=false))]
    fn from_dem(text: &str, enable_correlations: bool) -> PyResult<Self> {
        let dem = crate::dem_program::DemProgram::parse(text).map_err(err)?;
        let g = if enable_correlations { UserGraph::from_dem_with_correlations(&dem) } else { UserGraph::from_dem(&dem) };
        Ok(PyUserGraph { g: g.map_err(err)?, engine: Mutex::new(None) })
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

    /// PyMatching's `set_min_num_observables`, which does not set `_mwpm_needs_updating`: a kept
    /// decoder stays as built, so its answers fill only the fault ids it was built with (on
    /// ties, from its own path: by mask up to 64, by the search graph above), until a change to
    /// the graph or a `get_mwpm_with_search_graph` rebuilds it. Outputs are as long as the count
    /// is now.
    fn ensure_num_fault_ids(&mut self, n: usize) {
        self.g.ensure_num_fault_ids(n);
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

    /// Every weight a correlated load implied, by edge: (edge id, node1, node2 or None, weight).
    fn implied_weights(&self) -> Vec<(usize, usize, Option<usize>, f64)> {
        (0..self.g.num_edges()).flat_map(|id| self.g.implied_weights(id).iter().map(move |r| (id, r.node1, r.node2, r.weight))).collect()
    }

    /// Sets the weights edge `edge` implies for others: (node1, node2 or None, weight) each.
    fn set_implied_weights(&mut self, edge: usize, rules: Vec<(usize, Option<usize>, f64)>) -> PyResult<()> {
        let rules = rules.into_iter().map(|(node1, node2, weight)| ImpliedWeight { node1, node2, weight }).collect();
        self.g.set_implied_weights(edge, rules).map_err(err)?;
        self.changed();
        Ok(())
    }

    /// One shot (detection events as node indices): (flipped fault ids as one byte each,
    /// weight).
    fn decode<'py>(&self, py: Python<'py>, events: Vec<u64>) -> PyResult<(Bound<'py, PyBytes>, f64)> {
        // As long as the graph's count now (`get_num_observables`); the kept decoder fills its own.
        let nf = self.g.num_fault_ids();
        let (bytes, weight) = self.with_engine(py, false, |e| {
            let mut obs = vec![0u8; nf];
            let w = e.mwpm.decode_into(&events, &mut obs)?;
            Ok((obs, w as f64 / e.decoder.graph().normalising_constant))
        })?;
        Ok((PyBytes::new(py, &bytes), weight))
    }

    /// PyMatching's `get_mwpm_with_search_graph`, which its correlated calls make before they
    /// decode (here the correlated matcher decodes them): a kept decoder built without the
    /// search graph is rebuilt, with the fault-id count as it is now.
    fn mwpm_with_search_graph(&self, py: Python<'_>) -> PyResult<()> {
        self.with_engine(py, true, |_| Ok(()))
    }

    /// Shots as bit-packed rows of `row_bytes` bytes, bit k a detection event on node k:
    /// (predictions bit-packed per shot, `ceil(num_fault_ids / 8)` bytes each; weights as f64).
    /// Shots are split into contiguous runs, one per thread; a failure is the earliest shot's,
    /// PyMatching's message after `shot k: `.
    fn decode_batch<'py>(&self, py: Python<'py>, packed: &[u8], shots: usize, row_bytes: usize, threads: usize) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        if packed.len() != shots * row_bytes {
            return Err(err(format!("{} bytes of shots, not {shots} rows of {row_bytes}", packed.len())));
        }
        // Rows as long as the graph's count now; the kept decoder fills its own (see `decode`).
        let nf = self.g.num_fault_ids();
        let (preds, weights) = self.with_engine(py, false, |e| {
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
                                    // A padding bit past `num_nodes` is refused before any region is
                                    // made (`Mwpm::begin`), so the next decode works. Deliberately not
                                    // PyMatching's: it makes a region at each of the shot's earlier
                                    // detection events, then throws. When the shot has earlier
                                    // (non-boundary) detection events, every later decode on that mwpm
                                    // raises "!mwpm.flooder.queue.empty()" until a rebuild; with none,
                                    // its next decode works, as ours. (An event at a node a negative
                                    // edge marked only unmarks it, making no region; a region at a node
                                    // with no edges queues nothing, so only its next decode fails, with
                                    // "No perfect matching ...", and resets it.)
                                    *w = decoder.decode_packed_with(mwpm, &events, o).map_err(|e| format!("shot {shot}: {e}"))?;
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
        self.with_engine(py, true, |e| e.mwpm.try_decode_to_edges(&events))
    }

    /// The matched pairs of detection events (-1: the boundary), in PyMatching's order.
    fn decode_to_matched_pairs(&self, py: Python<'_>, events: Vec<u64>) -> PyResult<Vec<(i64, i64)>> {
        self.with_engine(py, false, |e| e.mwpm.try_decode_to_match_edges(&events))
    }
}

/// PyMatching's `animate` (`pm_exact::write_animated_decoding_svg_frames`): a model's graph as
/// PyMatching's animation builds it (1000 weight levels) and its detectors placed on the page.
#[pyclass(name = "PmAnimation", module = "stabilizer_qec._core")]
pub struct PyPmAnimation {
    graph: std::sync::Arc<crate::pm_exact::Graph>,
    coords: Vec<crate::pm_exact::Point>,
    boundary: Vec<crate::pm_exact::Point>,
}

#[pymethods]
impl PyPmAnimation {
    /// `dem_text`'s graph and `detector_coordinates` (each detector's, in order; empty for none)
    /// placed at `pixels_per_unit_length`, in PyMatching's order: the places, then the graph.
    #[new]
    fn new(dem_text: &str, detector_coordinates: Vec<Vec<f64>>, pixels_per_unit_length: u32) -> PyResult<Self> {
        use crate::pm_exact::{pick_coords_for_drawing, Graph, NeighbourOrder};
        let dem = crate::dem_program::DemProgram::parse(dem_text).map_err(err)?;
        let (coords, boundary) = pick_coords_for_drawing(&detector_coordinates, pixels_per_unit_length as f32);
        let g = UserGraph::from_dem(&dem).map_err(err)?;
        let graph = Graph::try_from_user_graph(&g, NeighbourOrder::V2_4, 1000).map_err(err)?;
        if coords.len() < graph.num_nodes {
            return Err(err(format!("{} detector coordinates for a graph of {} nodes", coords.len(), graph.num_nodes)));
        }
        Ok(PyPmAnimation { graph: std::sync::Arc::new(graph), coords, boundary })
    }

    /// Draws the run on `detection_events`, handing each frame (an SVG document) to `emit` as it
    /// is drawn. Raises what `emit` raises, and ValueError (PyMatching's `Unknown tentative event
    /// type.`) when the queue runs dry before every region is frozen.
    fn run(
        &self,
        detection_events: Vec<u64>,
        held_frames_per_event: usize,
        held_frames_at_start: usize,
        held_frames_at_end: usize,
        max_growth_between_frames: usize,
        emit: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        use crate::pm_exact::{write_animated_decoding_svg_frames, AnimationError, FrameSchedule, UNKNOWN_TENTATIVE_EVENT};
        if let Some(&d) = detection_events.iter().find(|&&d| d >= self.graph.num_nodes as u64) {
            return Err(err(format!("Detection event index {d} is too large for a graph of {} nodes", self.graph.num_nodes)));
        }
        let schedule = FrameSchedule {
            frames_per_mwpm_event: held_frames_per_event,
            hold_frames_at_start: held_frames_at_start,
            hold_frames_at_end: held_frames_at_end,
            max_growth_between_frames,
        };
        let mut m = Mwpm::new(self.graph.clone());
        match write_animated_decoding_svg_frames(&mut m, &self.coords, &self.boundary, &detection_events, schedule, |frame| emit.call1((frame,)).map(|_| ())) {
            Ok(()) => Ok(()),
            Err(AnimationError::Frame(e)) => Err(e),
            Err(AnimationError::UnknownTentativeEvent) => Err(err(UNKNOWN_TENTATIVE_EVENT.into())),
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyUserGraph>()?;
    m.add_class::<PyPmAnimation>()?;
    Ok(())
}

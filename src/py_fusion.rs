//! Python bindings for the Fusion Blossom port (`stabilizer_qec.fusion_blossom`): the classes
//! and functions of fusion_blossom 0.2.13's module, each a wrapper over its type in
//! `crate::fusion`, with the original's names, signatures, defaults and `repr`s. Its
//! assertions stay panics, raised in Python as they were in the original.

use std::collections::BTreeSet;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

use crate::fusion::dual_module::DualNodeClass;
use crate::fusion::example_codes::*;
use crate::fusion::mwpm_solver::*;
use crate::fusion::primal_module::{IntermediateMatching, PerfectMatching};
use crate::fusion::shim::serde_json::{self, FromJson, Number, ToJson, Value};
use crate::fusion::util::*;
use crate::fusion::visualize::{self, FusionVisualizer, VisualizePosition, Visualizer};

fn json_to_py(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    Ok(match value {
        Value::Null => py.None(),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any().unbind(),
        Value::Number(Number::PosInt(u)) => match i64::try_from(*u) {
            Ok(i) => i.into_pyobject(py)?.into_any().unbind(),
            Err(_) => (*u as f64).into_pyobject(py)?.into_any().unbind(),
        },
        Value::Number(Number::NegInt(i)) => i.into_pyobject(py)?.into_any().unbind(),
        Value::Number(Number::Float(x)) => x.into_pyobject(py)?.into_any().unbind(),
        Value::String(s) => PyString::new(py, s).into_any().unbind(),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_py(py, item)?)?;
            }
            list.into_any().unbind()
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (key, item) in map {
                dict.set_item(key, json_to_py(py, item)?)?;
            }
            dict.into_any().unbind()
        }
    })
}

fn py_to_json(object: &Bound<'_, PyAny>) -> PyResult<Value> {
    if object.is_none() {
        Ok(Value::Null)
    } else if object.is_instance_of::<PyBool>() {
        Ok(Value::Bool(object.extract()?))
    } else if object.is_instance_of::<PyInt>() {
        let i: i64 = object.extract()?;
        Ok(Value::Number(if i >= 0 { Number::PosInt(i as u64) } else { Number::NegInt(i) }))
    } else if object.is_instance_of::<PyFloat>() {
        Ok(Value::Number(Number::Float(object.extract()?)))
    } else if object.is_instance_of::<PyString>() {
        Ok(Value::String(object.extract()?))
    } else if object.is_instance_of::<PyList>() || object.is_instance_of::<PyTuple>() {
        object.try_iter()?.map(|item| py_to_json(&item?)).collect::<PyResult<Vec<_>>>().map(Value::Array)
    } else if let Ok(dict) = object.cast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (key, item) in dict.iter() {
            map.insert(key.extract::<String>()?, py_to_json(&item)?);
        }
        Ok(Value::Object(map))
    } else {
        Err(PyTypeError::new_err("unsupported python type, should be (cascaded) dict, list and basic numerical types"))
    }
}

fn to_json_text<T: ToJson>(value: &T) -> PyResult<String> {
    serde_json::to_string(value).map_err(PyTypeError::new_err)
}

fn from_json_text<T: FromJson>(text: &str) -> PyResult<T> {
    serde_json::from_str(text).map_err(PyTypeError::new_err)
}

#[pyclass(name = "SolverInitializer", module = "stabilizer_qec.fusion_blossom", from_py_object)]
#[derive(Clone)]
pub struct PySolverInitializer {
    inner: SolverInitializer,
}

#[pymethods]
impl PySolverInitializer {
    #[new]
    fn new(vertex_num: VertexNum, weighted_edges: Vec<(VertexIndex, VertexIndex, Weight)>, virtual_vertices: Vec<VertexIndex>) -> Self {
        Self { inner: SolverInitializer::new(vertex_num, weighted_edges, virtual_vertices) }
    }
    #[getter]
    fn vertex_num(&self) -> VertexNum {
        self.inner.vertex_num
    }
    #[setter]
    fn set_vertex_num(&mut self, value: VertexNum) {
        self.inner.vertex_num = value;
    }
    #[getter]
    fn weighted_edges(&self) -> Vec<(VertexIndex, VertexIndex, Weight)> {
        self.inner.weighted_edges.clone()
    }
    #[setter]
    fn set_weighted_edges(&mut self, value: Vec<(VertexIndex, VertexIndex, Weight)>) {
        self.inner.weighted_edges = value;
    }
    #[getter]
    fn virtual_vertices(&self) -> Vec<VertexIndex> {
        self.inner.virtual_vertices.clone()
    }
    #[setter]
    fn set_virtual_vertices(&mut self, value: Vec<VertexIndex>) {
        self.inner.virtual_vertices = value;
    }
    fn to_json(&self) -> PyResult<String> {
        to_json_text(&self.inner)
    }
    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        Ok(Self { inner: from_json_text(value)? })
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass(name = "SyndromePattern", module = "stabilizer_qec.fusion_blossom", from_py_object)]
#[derive(Clone)]
pub struct PySyndromePattern {
    inner: SyndromePattern,
}

#[pymethods]
impl PySyndromePattern {
    #[new]
    #[pyo3(signature = (defect_vertices=vec![], erasures=vec![], dynamic_weights=vec![], syndrome_vertices=None))]
    fn new(defect_vertices: Vec<VertexIndex>, erasures: Vec<EdgeIndex>, dynamic_weights: Vec<(EdgeIndex, Weight)>, syndrome_vertices: Option<Vec<VertexIndex>>) -> Self {
        Self { inner: SyndromePattern::py_new(defect_vertices, erasures, dynamic_weights, syndrome_vertices) }
    }
    #[staticmethod]
    fn new_vertices(defect_vertices: Vec<VertexIndex>) -> Self {
        Self { inner: SyndromePattern::new_vertices(defect_vertices) }
    }
    #[staticmethod]
    fn new_empty() -> Self {
        Self { inner: SyndromePattern::new_empty() }
    }
    #[getter]
    fn defect_vertices(&self) -> Vec<VertexIndex> {
        self.inner.defect_vertices.clone()
    }
    #[setter]
    fn set_defect_vertices(&mut self, value: Vec<VertexIndex>) {
        self.inner.defect_vertices = value;
    }
    #[getter]
    fn erasures(&self) -> Vec<EdgeIndex> {
        self.inner.erasures.clone()
    }
    #[setter]
    fn set_erasures(&mut self, value: Vec<EdgeIndex>) {
        self.inner.erasures = value;
    }
    #[getter]
    fn dynamic_weights(&self) -> Vec<(EdgeIndex, Weight)> {
        self.inner.dynamic_weights.clone()
    }
    #[setter]
    fn set_dynamic_weights(&mut self, value: Vec<(EdgeIndex, Weight)>) {
        self.inner.dynamic_weights = value;
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// `VertexRange`, `DefectRange`, `SyndromeRange` and `NodeRange`: one class, as in the original.
#[pyclass(name = "IndexRange", module = "stabilizer_qec.fusion_blossom", from_py_object)]
#[derive(Clone)]
pub struct PyIndexRange {
    inner: IndexRange,
}

#[pymethods]
impl PyIndexRange {
    #[new]
    fn new(start: VertexNodeIndex, end: VertexNodeIndex) -> Self {
        Self { inner: IndexRange::new(start, end) }
    }
    #[staticmethod]
    fn new_length(start: VertexNodeIndex, length: VertexNodeIndex) -> Self {
        Self { inner: IndexRange::new_length(start, length) }
    }
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn start(&self) -> VertexNodeIndex {
        self.inner.start()
    }
    fn end(&self) -> VertexNodeIndex {
        self.inner.end()
    }
    fn append_by(&mut self, append_count: VertexNodeIndex) {
        self.inner.append_by(append_count)
    }
    fn bias_by(&mut self, bias: VertexNodeIndex) {
        self.inner.bias_by(bias)
    }
    fn sanity_check(&self) {
        self.inner.sanity_check()
    }
    fn contains(&self, vertex_index: VertexNodeIndex) -> bool {
        self.inner.contains(vertex_index)
    }
    fn fuse(&self, other: PyRef<'_, Self>) -> (Self, Self) {
        let (whole, interface) = self.inner.fuse(&other.inner);
        (Self { inner: whole }, Self { inner: interface })
    }
    fn contains_any(&self, vertex_indices: Vec<VertexNodeIndex>) -> bool {
        self.inner.contains_any(&vertex_indices)
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

fn ranges(list: &[IndexRange]) -> Vec<PyIndexRange> {
    list.iter().map(|&inner| PyIndexRange { inner }).collect()
}

#[pyclass(name = "PartitionConfig", module = "stabilizer_qec.fusion_blossom", from_py_object)]
#[derive(Clone)]
pub struct PyPartitionConfig {
    inner: PartitionConfig,
}

#[pymethods]
impl PyPartitionConfig {
    #[new]
    fn new(vertex_num: VertexNum) -> Self {
        Self { inner: PartitionConfig::new(vertex_num) }
    }
    #[getter]
    fn vertex_num(&self) -> VertexNum {
        self.inner.vertex_num
    }
    #[setter]
    fn set_vertex_num(&mut self, value: VertexNum) {
        self.inner.vertex_num = value;
    }
    #[getter]
    fn partitions(&self) -> Vec<PyIndexRange> {
        ranges(&self.inner.partitions)
    }
    #[setter]
    fn set_partitions(&mut self, value: Vec<PyIndexRange>) {
        self.inner.partitions = value.into_iter().map(|r| r.inner).collect();
    }
    #[getter]
    fn fusions(&self) -> Vec<(usize, usize)> {
        self.inner.fusions.clone()
    }
    #[setter]
    fn set_fusions(&mut self, value: Vec<(usize, usize)>) {
        self.inner.fusions = value;
    }
    fn info(&self) -> PyPartitionInfo {
        PyPartitionInfo { inner: self.inner.info() }
    }
    fn to_json(&self) -> PyResult<String> {
        to_json_text(&self.inner)
    }
    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        Ok(Self { inner: from_json_text(value)? })
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass(name = "PartitionInfo", module = "stabilizer_qec.fusion_blossom", from_py_object)]
#[derive(Clone)]
pub struct PyPartitionInfo {
    inner: PartitionInfo,
}

#[pymethods]
impl PyPartitionInfo {
    #[getter]
    fn config(&self) -> PyPartitionConfig {
        PyPartitionConfig { inner: self.inner.config.clone() }
    }
    #[setter]
    fn set_config(&mut self, value: PyPartitionConfig) {
        self.inner.config = value.inner;
    }
    #[getter]
    fn units(&self) -> Vec<PyPartitionUnitInfo> {
        self.inner.units.iter().map(|u| PyPartitionUnitInfo { inner: u.clone() }).collect()
    }
    #[setter]
    fn set_units(&mut self, value: Vec<PyPartitionUnitInfo>) {
        self.inner.units = value.into_iter().map(|u| u.inner).collect();
    }
    #[getter]
    fn vertex_to_owning_unit(&self) -> Vec<usize> {
        self.inner.vertex_to_owning_unit.clone()
    }
    #[setter]
    fn set_vertex_to_owning_unit(&mut self, value: Vec<usize>) {
        self.inner.vertex_to_owning_unit = value;
    }
    fn partition_syndrome_unordered(&self, syndrome_pattern: PyRef<'_, PySyndromePattern>) -> Vec<PySyndromePattern> {
        self.inner.partition_syndrome_unordered(&syndrome_pattern.inner).into_iter().map(|inner| PySyndromePattern { inner }).collect()
    }
    fn to_json(&self) -> PyResult<String> {
        to_json_text(&self.inner)
    }
    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        Ok(Self { inner: from_json_text(value)? })
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass(name = "PartitionUnitInfo", module = "stabilizer_qec.fusion_blossom", from_py_object)]
#[derive(Clone)]
pub struct PyPartitionUnitInfo {
    inner: PartitionUnitInfo,
}

#[pymethods]
impl PyPartitionUnitInfo {
    #[getter]
    fn whole_range(&self) -> PyIndexRange {
        PyIndexRange { inner: self.inner.whole_range }
    }
    #[setter]
    fn set_whole_range(&mut self, value: PyIndexRange) {
        self.inner.whole_range = value.inner;
    }
    #[getter]
    fn owning_range(&self) -> PyIndexRange {
        PyIndexRange { inner: self.inner.owning_range }
    }
    #[setter]
    fn set_owning_range(&mut self, value: PyIndexRange) {
        self.inner.owning_range = value.inner;
    }
    #[getter]
    fn children(&self) -> Option<(usize, usize)> {
        self.inner.children
    }
    #[setter]
    fn set_children(&mut self, value: Option<(usize, usize)>) {
        self.inner.children = value;
    }
    #[getter]
    fn parent(&self) -> Option<usize> {
        self.inner.parent
    }
    #[setter]
    fn set_parent(&mut self, value: Option<usize>) {
        self.inner.parent = value;
    }
    #[getter]
    fn leaves(&self) -> Vec<usize> {
        self.inner.leaves.clone()
    }
    #[setter]
    fn set_leaves(&mut self, value: Vec<usize>) {
        self.inner.leaves = value;
    }
    #[getter]
    fn descendants(&self) -> BTreeSet<usize> {
        self.inner.descendants.clone()
    }
    #[setter]
    fn set_descendants(&mut self, value: BTreeSet<usize>) {
        self.inner.descendants = value;
    }
    fn to_json(&self) -> PyResult<String> {
        to_json_text(&self.inner)
    }
    #[staticmethod]
    fn from_json(value: &str) -> PyResult<Self> {
        Ok(Self { inner: from_json_text(value)? })
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass(name = "VisualizePosition", module = "stabilizer_qec.fusion_blossom", from_py_object)]
#[derive(Clone)]
pub struct PyVisualizePosition {
    inner: VisualizePosition,
}

#[pymethods]
impl PyVisualizePosition {
    #[new]
    fn new(i: f64, j: f64, t: f64) -> Self {
        Self { inner: VisualizePosition::new(i, j, t) }
    }
    #[getter]
    fn i(&self) -> f64 {
        self.inner.i
    }
    #[setter]
    fn set_i(&mut self, value: f64) {
        self.inner.i = value;
    }
    #[getter]
    fn j(&self) -> f64 {
        self.inner.j
    }
    #[setter]
    fn set_j(&mut self, value: f64) {
        self.inner.j = value;
    }
    #[getter]
    fn t(&self) -> f64 {
        self.inner.t
    }
    #[setter]
    fn set_t(&mut self, value: f64) {
        self.inner.t = value;
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

fn positions_in(list: Vec<PyVisualizePosition>) -> Vec<VisualizePosition> {
    list.into_iter().map(|p| p.inner).collect()
}

fn positions_out(list: Vec<VisualizePosition>) -> Vec<PyVisualizePosition> {
    list.into_iter().map(|inner| PyVisualizePosition { inner }).collect()
}

#[pyclass(name = "Visualizer", module = "stabilizer_qec.fusion_blossom")]
pub struct PyVisualizer {
    inner: Visualizer,
}

fn snapshot_of(object: &Bound<'_, PyAny>) -> PyResult<Value> {
    py_to_json(&object.call_method0("snapshot")?)
}

#[pymethods]
impl PyVisualizer {
    #[new]
    #[pyo3(signature = (filepath, positions=vec![], center=true))]
    fn new(filepath: Option<String>, positions: Vec<PyVisualizePosition>, center: bool) -> PyResult<Self> {
        Ok(Self { inner: Visualizer::new(filepath, positions_in(positions), center)? })
    }
    #[getter]
    fn snapshots(&self) -> Vec<String> {
        self.inner.snapshots.clone()
    }
    fn snapshot_combined(&mut self, name: String, object_pys: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        let values = object_pys.iter().map(snapshot_of).collect::<PyResult<Vec<_>>>()?;
        Ok(self.inner.snapshot_combined_value(name, values)?)
    }
    fn snapshot(&mut self, name: String, object_py: Bound<'_, PyAny>) -> PyResult<()> {
        let value = snapshot_of(&object_py)?;
        Ok(self.inner.snapshot_value(name, value)?)
    }
    fn snapshot_combined_value(&mut self, name: String, value_pys: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        let values = value_pys.iter().map(py_to_json).collect::<PyResult<Vec<_>>>()?;
        Ok(self.inner.snapshot_combined_value(name, values)?)
    }
    fn snapshot_value(&mut self, name: String, value_py: Bound<'_, PyAny>) -> PyResult<()> {
        let value = py_to_json(&value_py)?;
        Ok(self.inner.snapshot_value(name, value)?)
    }
}

#[pyclass(name = "PerfectMatching", module = "stabilizer_qec.fusion_blossom")]
pub struct PyPerfectMatching {
    inner: PerfectMatching,
}

#[pymethods]
impl PyPerfectMatching {
    #[new]
    fn new() -> Self {
        Self { inner: PerfectMatching::new() }
    }
    fn legacy_get_mwpm_result(&self, defect_vertices: Vec<VertexIndex>) -> Vec<DefectIndex> {
        self.inner.legacy_get_mwpm_result(defect_vertices)
    }
    #[getter]
    fn peer_matchings(&self) -> Vec<(NodeIndex, NodeIndex)> {
        self.inner.peer_matchings.iter().map(|(a, b)| (a.updated_index(), b.updated_index())).collect()
    }
    #[getter]
    fn virtual_matchings(&self) -> Vec<(NodeIndex, VertexIndex)> {
        self.inner.virtual_matchings.iter().map(|(a, b)| (a.updated_index(), *b)).collect()
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass(name = "IntermediateMatching", module = "stabilizer_qec.fusion_blossom")]
pub struct PyIntermediateMatching {
    inner: IntermediateMatching,
}

#[pymethods]
impl PyIntermediateMatching {
    #[new]
    fn new() -> Self {
        Self { inner: IntermediateMatching::new() }
    }
    fn append(&mut self, mut other: PyRefMut<'_, Self>) {
        self.inner.append(&mut other.inner)
    }
    fn get_perfect_matching(&self) -> PyPerfectMatching {
        PyPerfectMatching { inner: self.inner.get_perfect_matching() }
    }
    #[getter]
    fn peer_matchings(&self) -> Vec<((NodeIndex, NodeIndex), (NodeIndex, NodeIndex))> {
        self.inner
            .peer_matchings
            .iter()
            .map(|((a, b), (c, d))| ((a.updated_index(), b.upgrade_force().updated_index()), (c.updated_index(), d.upgrade_force().updated_index())))
            .collect()
    }
    #[getter]
    fn virtual_matchings(&self) -> Vec<((NodeIndex, NodeIndex), VertexIndex)> {
        self.inner.virtual_matchings.iter().map(|((a, b), c)| ((a.updated_index(), b.upgrade_force().updated_index()), *c)).collect()
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// The methods every solver has in Python (the original's `bind_trait_primal_dual_solver`),
/// with its own constructor and extras.
macro_rules! solver_class {
    ($py:ident, $inner:ty, $name:literal, { $($extra:tt)* }) => {
        #[pyclass(name = $name, module = "stabilizer_qec.fusion_blossom")]
        pub struct $py {
            inner: $inner,
        }

        #[pymethods]
        impl $py {
            $($extra)*

            fn clear(&mut self) {
                self.inner.clear()
            }
            #[pyo3(signature = (syndrome_pattern, visualizer=None))]
            fn solve_visualizer(&mut self, syndrome_pattern: PyRef<'_, PySyndromePattern>, mut visualizer: Option<PyRefMut<'_, PyVisualizer>>) {
                self.inner.solve_visualizer(&syndrome_pattern.inner, visualizer.as_deref_mut().map(|v| &mut v.inner))
            }
            #[pyo3(signature = (syndrome_pattern, visualizer=None))]
            fn solve(&mut self, syndrome_pattern: PyRef<'_, PySyndromePattern>, visualizer: Option<PyRefMut<'_, PyVisualizer>>) {
                self.solve_visualizer(syndrome_pattern, visualizer)
            }
            #[pyo3(signature = (visualizer=None))]
            fn perfect_matching_visualizer(&mut self, mut visualizer: Option<PyRefMut<'_, PyVisualizer>>) -> PyPerfectMatching {
                PyPerfectMatching { inner: self.inner.perfect_matching_visualizer(visualizer.as_deref_mut().map(|v| &mut v.inner)) }
            }
            #[pyo3(signature = (visualizer=None))]
            fn perfect_matching(&mut self, visualizer: Option<PyRefMut<'_, PyVisualizer>>) -> PyPerfectMatching {
                self.perfect_matching_visualizer(visualizer)
            }
            #[pyo3(signature = (visualizer=None))]
            fn subgraph_visualizer(&mut self, mut visualizer: Option<PyRefMut<'_, PyVisualizer>>) -> Vec<EdgeIndex> {
                self.inner.subgraph_visualizer(visualizer.as_deref_mut().map(|v| &mut v.inner))
            }
            #[pyo3(signature = (visualizer=None))]
            fn subgraph(&mut self, visualizer: Option<PyRefMut<'_, PyVisualizer>>) -> Vec<EdgeIndex> {
                self.subgraph_visualizer(visualizer)
            }
            fn sum_dual_variables(&self) -> Weight {
                self.inner.sum_dual_variables()
            }
            fn generate_profiler_report(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                json_to_py(py, &self.inner.generate_profiler_report())
            }
            fn stim_integration_predict_bit_packed_data(&mut self, in_file: String, out_file: String, edge_masks: Vec<usize>, num_shots: usize, num_dets: usize, num_obs: usize) {
                self.inner.stim_integration_predict_bit_packed_data(in_file, out_file, &edge_masks, num_shots, num_dets, num_obs)
            }
        }
    };
}

solver_class!(PySolverSerial, SolverSerial, "SolverSerial", {
    #[pyo3(signature = (abbrev=true))]
    fn snapshot(&self, py: Python<'_>, abbrev: bool) -> PyResult<Py<PyAny>> {
        json_to_py(py, &self.inner.snapshot(abbrev))
    }

    #[new]
    #[pyo3(signature = (initializer, *, max_tree_size=None))]
    fn new(initializer: PyRef<'_, PySolverInitializer>, max_tree_size: Option<usize>) -> Self {
        let solver = SolverSerial::new(&initializer.inner);
        if let Some(max_tree_size) = max_tree_size {
            solver.primal_module.write().max_tree_size = max_tree_size;
        }
        Self { inner: solver }
    }
});

solver_class!(PySolverDualParallel, SolverDualParallel, "SolverDualParallel", {
    #[pyo3(signature = (abbrev=true))]
    fn snapshot(&self, py: Python<'_>, abbrev: bool) -> PyResult<Py<PyAny>> {
        json_to_py(py, &self.inner.snapshot(abbrev))
    }

    #[new]
    fn new(initializer: PyRef<'_, PySolverInitializer>, partition_info: PyRef<'_, PyPartitionInfo>, primal_dual_config: Bound<'_, PyAny>) -> PyResult<Self> {
        let config = py_to_json(&primal_dual_config)?;
        Ok(Self { inner: SolverDualParallel::new(&initializer.inner, &partition_info.inner, config) })
    }
});

solver_class!(PySolverParallel, SolverParallel, "SolverParallel", {
    #[pyo3(signature = (abbrev=true))]
    fn snapshot(&self, py: Python<'_>, abbrev: bool) -> PyResult<Py<PyAny>> {
        json_to_py(py, &self.inner.snapshot(abbrev))
    }

    #[new]
    fn new(initializer: PyRef<'_, PySolverInitializer>, partition_info: PyRef<'_, PyPartitionInfo>, primal_dual_config: Bound<'_, PyAny>) -> PyResult<Self> {
        let config = py_to_json(&primal_dual_config)?;
        Ok(Self { inner: SolverParallel::new(&initializer.inner, &partition_info.inner, config) })
    }

    fn defect_perfect_matching(&mut self) -> Vec<(VertexIndex, VertexIndex)> {
        let defect_of = |ptr: &crate::fusion::dual_module::DualNodePtr| {
            let node = ptr.read_recursive();
            if let DualNodeClass::DefectVertex { defect_index } = &node.class {
                *defect_index
            } else {
                unreachable!("can only be syndrome")
            }
        };
        let perfect_matching = self.inner.perfect_matching_visualizer(None);
        let mut defect_matching = vec![];
        for (a, b) in perfect_matching.peer_matchings.iter() {
            defect_matching.push((defect_of(a), defect_of(b)));
        }
        for (a, virtual_vertex) in perfect_matching.virtual_matchings.iter() {
            defect_matching.push((defect_of(a), *virtual_vertex));
        }
        defect_matching
    }
});

// No constructor in Python, as in the original.
solver_class!(PySolverErrorPatternLogger, SolverErrorPatternLogger, "SolverErrorPatternLogger", {});

/// The original's serial solver of an earlier interface: in Python only its initializer, and
/// no constructor.
#[pyclass(name = "LegacySolverSerial", module = "stabilizer_qec.fusion_blossom")]
pub struct PyLegacySolverSerial {
    inner: LegacySolverSerial,
}

#[pymethods]
impl PyLegacySolverSerial {
    #[getter]
    fn initializer(&self) -> PySolverInitializer {
        PySolverInitializer { inner: self.inner.initializer.clone() }
    }
    #[setter]
    fn set_initializer(&mut self, value: PySolverInitializer) {
        self.inner.initializer = value.inner;
    }
}

#[pyclass(name = "CodeVertex", module = "stabilizer_qec.fusion_blossom", from_py_object)]
#[derive(Clone)]
pub struct PyCodeVertex {
    inner: CodeVertex,
}

#[pymethods]
impl PyCodeVertex {
    #[getter]
    fn position(&self) -> PyVisualizePosition {
        PyVisualizePosition { inner: self.inner.position.clone() }
    }
    #[setter]
    fn set_position(&mut self, value: PyVisualizePosition) {
        self.inner.position = value.inner;
    }
    #[getter]
    fn neighbor_edges(&self) -> Vec<EdgeIndex> {
        self.inner.neighbor_edges.clone()
    }
    #[setter]
    fn set_neighbor_edges(&mut self, value: Vec<EdgeIndex>) {
        self.inner.neighbor_edges = value;
    }
    #[getter]
    fn is_virtual(&self) -> bool {
        self.inner.is_virtual
    }
    #[setter]
    fn set_is_virtual(&mut self, value: bool) {
        self.inner.is_virtual = value;
    }
    #[getter]
    fn is_defect(&self) -> bool {
        self.inner.is_defect
    }
    #[setter]
    fn set_is_defect(&mut self, value: bool) {
        self.inner.is_defect = value;
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[pyclass(name = "CodeEdge", module = "stabilizer_qec.fusion_blossom", from_py_object)]
#[derive(Clone)]
pub struct PyCodeEdge {
    inner: CodeEdge,
}

#[pymethods]
impl PyCodeEdge {
    #[new]
    fn new(a: VertexIndex, b: VertexIndex) -> Self {
        Self { inner: CodeEdge::new(a, b) }
    }
    #[getter]
    fn vertices(&self) -> (VertexIndex, VertexIndex) {
        self.inner.vertices
    }
    #[setter]
    fn set_vertices(&mut self, value: (VertexIndex, VertexIndex)) {
        self.inner.vertices = value;
    }
    #[getter]
    fn p(&self) -> f64 {
        self.inner.p
    }
    #[setter]
    fn set_p(&mut self, value: f64) {
        self.inner.p = value;
    }
    #[getter]
    fn pe(&self) -> f64 {
        self.inner.pe
    }
    #[setter]
    fn set_pe(&mut self, value: f64) {
        self.inner.pe = value;
    }
    #[getter]
    fn half_weight(&self) -> Weight {
        self.inner.half_weight
    }
    #[setter]
    fn set_half_weight(&mut self, value: Weight) {
        self.inner.half_weight = value;
    }
    #[getter]
    fn is_erasure(&self) -> bool {
        self.inner.is_erasure
    }
    #[setter]
    fn set_is_erasure(&mut self, value: bool) {
        self.inner.is_erasure = value;
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// An example code's class: the original's `bind_trait_example_code` methods, the vertices
/// and edges, and its own constructors and fields.
macro_rules! code_class {
    ($py:ident, $inner:ty, $name:literal, { $($extra:tt)* }) => {
        #[pyclass(name = $name, module = "stabilizer_qec.fusion_blossom", from_py_object)]
        #[derive(Clone)]
        pub struct $py {
            inner: $inner,
        }

        #[pymethods]
        impl $py {
            $($extra)*

            fn __repr__(&self) -> String {
                format!("{:?}", self.inner)
            }
            #[getter]
            fn vertices(&self) -> Vec<PyCodeVertex> {
                self.inner.vertices.iter().map(|v| PyCodeVertex { inner: v.clone() }).collect()
            }
            #[setter]
            fn set_vertices(&mut self, value: Vec<PyCodeVertex>) {
                self.inner.vertices = value.into_iter().map(|v| v.inner).collect();
            }
            #[getter]
            fn edges(&self) -> Vec<PyCodeEdge> {
                self.inner.edges.iter().map(|e| PyCodeEdge { inner: e.clone() }).collect()
            }
            #[setter]
            fn set_edges(&mut self, value: Vec<PyCodeEdge>) {
                self.inner.edges = value.into_iter().map(|e| e.inner).collect();
            }
            fn vertex_num(&self) -> VertexNum {
                self.inner.vertex_num()
            }
            fn compute_weights(&mut self, max_half_weight: Weight) {
                self.inner.compute_weights(max_half_weight)
            }
            fn sanity_check(&self) -> Option<String> {
                self.inner.sanity_check().err()
            }
            fn set_probability(&mut self, p: f64) {
                self.inner.set_probability(p)
            }
            fn set_erasure_probability(&mut self, p: f64) {
                self.inner.set_erasure_probability(p)
            }
            fn fill_vertices(&mut self, vertex_num: VertexNum) {
                self.inner.fill_vertices(vertex_num)
            }
            fn get_positions(&self) -> Vec<PyVisualizePosition> {
                positions_out(self.inner.get_positions())
            }
            fn get_initializer(&self) -> PySolverInitializer {
                PySolverInitializer { inner: self.inner.get_initializer() }
            }
            fn set_defect_vertices(&mut self, defect_vertices: Vec<VertexIndex>) {
                self.inner.set_defect_vertices(&defect_vertices)
            }
            fn set_erasures(&mut self, erasures: Vec<EdgeIndex>) {
                self.inner.set_erasures(&erasures)
            }
            fn set_syndrome(&mut self, syndrome_pattern: PyRef<'_, PySyndromePattern>) {
                self.inner.set_syndrome(&syndrome_pattern.inner)
            }
            fn get_defect_vertices(&self) -> Vec<VertexIndex> {
                self.inner.get_defect_vertices()
            }
            fn get_erasures(&self) -> Vec<EdgeIndex> {
                self.inner.get_erasures()
            }
            fn get_syndrome(&self) -> PySyndromePattern {
                PySyndromePattern { inner: self.inner.get_syndrome() }
            }
            /// Without a seed, a random one (the original drew it from `thread_rng`).
            #[pyo3(signature = (seed=None))]
            fn generate_random_errors(&mut self, seed: Option<u64>) -> PySyndromePattern {
                PySyndromePattern { inner: self.inner.generate_random_errors(seed.unwrap_or_else(rand::random)) }
            }
            fn generate_errors(&mut self, edge_indices: Vec<EdgeIndex>) -> PySyndromePattern {
                PySyndromePattern { inner: self.inner.generate_errors(&edge_indices) }
            }
            fn clear_errors(&mut self) {
                self.inner.clear_errors()
            }
            fn is_virtual(&self, vertex_idx: usize) -> bool {
                self.inner.is_virtual(vertex_idx)
            }
            fn is_defect(&self, vertex_idx: usize) -> bool {
                self.inner.is_defect(vertex_idx)
            }
            fn reorder_vertices(&mut self, sequential_vertices: Vec<VertexIndex>) {
                self.inner.reorder_vertices(&sequential_vertices)
            }
            #[pyo3(signature = (abbrev=true))]
            fn snapshot(&self, py: Python<'_>, abbrev: bool) -> PyResult<Py<PyAny>> {
                json_to_py(py, &self.inner.snapshot(abbrev))
            }
        }
    };
}

code_class!(PyCodeCapacityRepetitionCode, CodeCapacityRepetitionCode, "CodeCapacityRepetitionCode", {
    #[new]
    #[pyo3(signature = (d, p, max_half_weight=500))]
    fn new(d: VertexNum, p: f64, max_half_weight: Weight) -> Self {
        Self { inner: CodeCapacityRepetitionCode::new(d, p, max_half_weight) }
    }
    #[staticmethod]
    fn create_code(d: VertexNum) -> Self {
        Self { inner: CodeCapacityRepetitionCode::create_code(d) }
    }
});

code_class!(PyCodeCapacityPlanarCode, CodeCapacityPlanarCode, "CodeCapacityPlanarCode", {
    #[new]
    #[pyo3(signature = (d, p, max_half_weight=500))]
    fn new(d: VertexNum, p: f64, max_half_weight: Weight) -> Self {
        Self { inner: CodeCapacityPlanarCode::new(d, p, max_half_weight) }
    }
    #[staticmethod]
    fn create_code(d: VertexNum) -> Self {
        Self { inner: CodeCapacityPlanarCode::create_code(d) }
    }
});

code_class!(PyPhenomenologicalPlanarCode, PhenomenologicalPlanarCode, "PhenomenologicalPlanarCode", {
    #[new]
    #[pyo3(signature = (d, noisy_measurements, p, max_half_weight=500))]
    fn new(d: VertexNum, noisy_measurements: VertexNum, p: f64, max_half_weight: Weight) -> Self {
        Self { inner: PhenomenologicalPlanarCode::new(d, noisy_measurements, p, max_half_weight) }
    }
    #[staticmethod]
    fn create_code(d: VertexNum, noisy_measurements: VertexNum) -> Self {
        Self { inner: PhenomenologicalPlanarCode::create_code(d, noisy_measurements) }
    }
});

code_class!(PyCircuitLevelPlanarCode, CircuitLevelPlanarCode, "CircuitLevelPlanarCode", {
    #[new]
    #[pyo3(signature = (d, noisy_measurements, p, max_half_weight=500))]
    fn new(d: VertexNum, noisy_measurements: VertexNum, p: f64, max_half_weight: Weight) -> Self {
        Self { inner: CircuitLevelPlanarCode::new(d, noisy_measurements, p, max_half_weight) }
    }
    #[staticmethod]
    #[pyo3(signature = (d, noisy_measurements, p, max_half_weight=500, diagonal_p=None))]
    fn new_diagonal(d: VertexNum, noisy_measurements: VertexNum, p: f64, max_half_weight: Weight, diagonal_p: Option<f64>) -> Self {
        Self { inner: CircuitLevelPlanarCode::new_diagonal(d, noisy_measurements, p, max_half_weight, diagonal_p) }
    }
    #[staticmethod]
    fn create_code(d: VertexNum, noisy_measurements: VertexNum) -> Self {
        Self { inner: CircuitLevelPlanarCode::create_code(d, noisy_measurements) }
    }
});

code_class!(PyCodeCapacityRotatedCode, CodeCapacityRotatedCode, "CodeCapacityRotatedCode", {
    #[new]
    #[pyo3(signature = (d, p, max_half_weight=500))]
    fn new(d: VertexNum, p: f64, max_half_weight: Weight) -> Self {
        Self { inner: CodeCapacityRotatedCode::new(d, p, max_half_weight) }
    }
    #[staticmethod]
    fn create_code(d: VertexNum) -> Self {
        Self { inner: CodeCapacityRotatedCode::create_code(d) }
    }
});

code_class!(PyPhenomenologicalRotatedCode, PhenomenologicalRotatedCode, "PhenomenologicalRotatedCode", {
    #[new]
    #[pyo3(signature = (d, noisy_measurements, p, max_half_weight=500))]
    fn new(d: VertexNum, noisy_measurements: VertexNum, p: f64, max_half_weight: Weight) -> Self {
        Self { inner: PhenomenologicalRotatedCode::new(d, noisy_measurements, p, max_half_weight) }
    }
    #[staticmethod]
    fn create_code(d: VertexNum, noisy_measurements: VertexNum) -> Self {
        Self { inner: PhenomenologicalRotatedCode::create_code(d, noisy_measurements) }
    }
});

code_class!(PyErrorPatternReader, ErrorPatternReader, "ErrorPatternReader", {
    #[new]
    #[pyo3(signature = (filename, cyclic_syndrome=false))]
    fn new(filename: String, cyclic_syndrome: bool) -> Self {
        Self { inner: ErrorPatternReader::py_new(filename, cyclic_syndrome) }
    }
    #[getter]
    fn syndrome_patterns(&self) -> Vec<PySyndromePattern> {
        self.inner.syndrome_patterns.iter().map(|s| PySyndromePattern { inner: s.clone() }).collect()
    }
    #[setter]
    fn set_syndrome_patterns(&mut self, value: Vec<PySyndromePattern>) {
        self.inner.syndrome_patterns = value.into_iter().map(|s| s.inner).collect();
    }
    #[getter]
    fn defect_index(&self) -> usize {
        self.inner.defect_index
    }
    #[setter]
    fn set_defect_index(&mut self, value: usize) {
        self.inner.defect_index = value;
    }
    #[getter]
    fn cyclic_syndrome(&self) -> bool {
        self.inner.cyclic_syndrome
    }
    #[setter]
    fn set_cyclic_syndrome(&mut self, value: bool) {
        self.inner.cyclic_syndrome = value;
    }
});

#[pyfunction(name = "weight_of_p")]
fn py_weight_of_p(p: f64) -> f64 {
    crate::fusion::example_codes::weight_of_p(p)
}

#[pyfunction]
fn center_positions(positions: Vec<PyVisualizePosition>) -> Vec<PyVisualizePosition> {
    positions_out(visualize::center_positions(positions_in(positions)))
}

#[pyfunction]
fn static_visualize_data_filename() -> String {
    visualize::static_visualize_data_filename()
}

#[pyfunction]
fn auto_visualize_data_filename() -> String {
    visualize::auto_visualize_data_filename()
}

#[pyfunction]
fn print_visualize_link_with_parameters(filename: String, parameters: Vec<(String, String)>) {
    visualize::print_visualize_link_with_parameters(filename, parameters)
}

#[pyfunction]
fn print_visualize_link(filename: String) {
    visualize::print_visualize_link(filename)
}

/// `_core.fusion_blossom`, the module `stabilizer_qec.fusion_blossom` re-exports.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let fb = PyModule::new(py, "fusion_blossom")?;
    fb.add_class::<PySolverInitializer>()?;
    fb.add_class::<PySyndromePattern>()?;
    fb.add_class::<PyPartitionConfig>()?;
    fb.add_class::<PyPartitionInfo>()?;
    fb.add_class::<PyPartitionUnitInfo>()?;
    let range = py.get_type::<PyIndexRange>();
    for name in ["VertexRange", "DefectRange", "SyndromeRange", "NodeRange"] {
        fb.add(name, &range)?;
    }
    fb.add_class::<PyLegacySolverSerial>()?;
    fb.add_class::<PySolverSerial>()?;
    fb.add_class::<PySolverDualParallel>()?;
    fb.add_class::<PySolverParallel>()?;
    fb.add_class::<PySolverErrorPatternLogger>()?;
    fb.add_class::<PyCodeVertex>()?;
    fb.add_class::<PyCodeEdge>()?;
    fb.add_function(wrap_pyfunction!(py_weight_of_p, &fb)?)?;
    fb.add_class::<PyCodeCapacityRepetitionCode>()?;
    fb.add_class::<PyCodeCapacityPlanarCode>()?;
    fb.add_class::<PyPhenomenologicalPlanarCode>()?;
    fb.add_class::<PyCircuitLevelPlanarCode>()?;
    fb.add_class::<PyCodeCapacityRotatedCode>()?;
    fb.add_class::<PyPhenomenologicalRotatedCode>()?;
    fb.add_class::<PyErrorPatternReader>()?;
    fb.add_class::<PyVisualizePosition>()?;
    fb.add_class::<PyVisualizer>()?;
    fb.add_function(wrap_pyfunction!(static_visualize_data_filename, &fb)?)?;
    fb.add_function(wrap_pyfunction!(auto_visualize_data_filename, &fb)?)?;
    fb.add_function(wrap_pyfunction!(print_visualize_link_with_parameters, &fb)?)?;
    fb.add_function(wrap_pyfunction!(print_visualize_link, &fb)?)?;
    fb.add_function(wrap_pyfunction!(center_positions, &fb)?)?;
    fb.add_class::<PyIntermediateMatching>()?;
    fb.add_class::<PyPerfectMatching>()?;
    m.add_submodule(&fb)?;
    Ok(())
}

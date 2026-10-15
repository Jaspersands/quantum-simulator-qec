//! Fusion Blossom's `serde` derives, written out: each type reads and writes the JSON its
//! derive did (field names, `#[serde(default)]`s, `deny_unknown_fields` where it had it,
//! `IndexRange` transparent as its two-element array). `to_json` sorts the keys, as
//! `serde_json::to_value` does; `write_text` keeps the fields' order, as `serde_json::to_string`
//! does.

use super::dual_module_parallel::{dual_module_parallel_default_configs as dual_defaults, DualModuleParallelConfig};
use super::primal_module_parallel::{primal_module_parallel_default_configs as primal_defaults, PrimalModuleParallelConfig, PrimalModuleParallelUnitEventTime};
use super::shim::serde_json::{write_fields, Error, Fields, FromJson, ToJson, Value};
use super::util::*;
use super::visualize::VisualizePosition;

macro_rules! object {
    ($($key:literal: $value:expr),* $(,)?) => {{
        let mut m = super::shim::serde_json::Map::new();
        $( m.insert($key.to_string(), ToJson::to_json(&$value)); )*
        Value::Object(m)
    }};
}

/// `ToJson` for a struct: its fields, as an object (`to_json`) and, in order, as text.
macro_rules! struct_json {
    ($t:ty, $self_:ident => { $($key:literal: $value:expr),* $(,)? }) => {
        impl ToJson for $t {
            fn to_json(&$self_) -> Value {
                object! { $($key: $value),* }
            }
            fn write_text(&$self_, out: &mut String) {
                write_fields(&[$(($key, &$value as &dyn ToJson)),*], out)
            }
        }
    };
}

impl ToJson for IndexRange {
    fn to_json(&self) -> Value {
        self.range.to_json()
    }
    fn write_text(&self, out: &mut String) {
        self.range.write_text(out)
    }
}

impl FromJson for IndexRange {
    fn from_json(value: &Value) -> Result<Self, Error> {
        let (a, b) = <(VertexNodeIndex, VertexNodeIndex)>::from_json(value)?;
        Ok(IndexRange { range: [a, b] })
    }
}

struct_json!(SolverInitializer, self => { "vertex_num": self.vertex_num, "weighted_edges": self.weighted_edges, "virtual_vertices": self.virtual_vertices });

impl FromJson for SolverInitializer {
    fn from_json(value: &Value) -> Result<Self, Error> {
        let mut f = Fields::of(value, "SolverInitializer")?;
        let out = SolverInitializer { vertex_num: f.field("vertex_num", None)?, weighted_edges: f.field("weighted_edges", None)?, virtual_vertices: f.field("virtual_vertices", None)? };
        f.finish(false)?;
        Ok(out)
    }
}

struct_json!(SyndromePattern, self => { "defect_vertices": self.defect_vertices, "erasures": self.erasures, "dynamic_weights": self.dynamic_weights });

impl FromJson for SyndromePattern {
    fn from_json(value: &Value) -> Result<Self, Error> {
        let mut f = Fields::of(value, "SyndromePattern")?;
        let out = SyndromePattern {
            defect_vertices: f.field("defect_vertices", None)?,
            erasures: f.field("erasures", Some(default_erasures))?,
            dynamic_weights: f.field("dynamic_weights", Some(default_dynamic_weights))?,
        };
        f.finish(false)?;
        Ok(out)
    }
}

struct_json!(PartitionConfig, self => { "vertex_num": self.vertex_num, "partitions": self.partitions, "fusions": self.fusions });

impl FromJson for PartitionConfig {
    fn from_json(value: &Value) -> Result<Self, Error> {
        let mut f = Fields::of(value, "PartitionConfig")?;
        let out = PartitionConfig { vertex_num: f.field("vertex_num", None)?, partitions: f.field("partitions", None)?, fusions: f.field("fusions", None)? };
        f.finish(true)?;
        Ok(out)
    }
}

struct_json!(PartitionUnitInfo, self => { "whole_range": self.whole_range, "owning_range": self.owning_range, "children": self.children, "parent": self.parent, "leaves": self.leaves, "descendants": self.descendants, });

impl FromJson for PartitionUnitInfo {
    fn from_json(value: &Value) -> Result<Self, Error> {
        let mut f = Fields::of(value, "PartitionUnitInfo")?;
        let out = PartitionUnitInfo {
            whole_range: f.field("whole_range", None)?,
            owning_range: f.field("owning_range", None)?,
            children: f.optional("children")?,
            parent: f.optional("parent")?,
            leaves: f.field("leaves", None)?,
            descendants: f.field::<Vec<usize>>("descendants", None)?.into_iter().collect(),
        };
        f.finish(false)?;
        Ok(out)
    }
}

struct_json!(PartitionInfo, self => { "config": self.config, "units": self.units, "vertex_to_owning_unit": self.vertex_to_owning_unit });

impl FromJson for PartitionInfo {
    fn from_json(value: &Value) -> Result<Self, Error> {
        let mut f = Fields::of(value, "PartitionInfo")?;
        let out = PartitionInfo { config: f.field("config", None)?, units: f.field("units", None)?, vertex_to_owning_unit: f.field("vertex_to_owning_unit", None)? };
        f.finish(false)?;
        Ok(out)
    }
}

struct_json!(VisualizePosition, self => { "i": self.i, "j": self.j, "t": self.t });

impl FromJson for VisualizePosition {
    fn from_json(value: &Value) -> Result<Self, Error> {
        let mut f = Fields::of(value, "VisualizePosition")?;
        let out = VisualizePosition { i: f.field("i", None)?, j: f.field("j", None)?, t: f.field("t", None)? };
        f.finish(false)?;
        Ok(out)
    }
}

struct_json!(DualModuleParallelConfig, self => { "thread_pool_size": self.thread_pool_size, "edges_in_fusion_unit": self.edges_in_fusion_unit, "enable_parallel_execution": self.enable_parallel_execution, });

impl FromJson for DualModuleParallelConfig {
    fn from_json(value: &Value) -> Result<Self, Error> {
        let mut f = Fields::of(value, "DualModuleParallelConfig")?;
        let out = DualModuleParallelConfig {
            thread_pool_size: f.field("thread_pool_size", Some(dual_defaults::thread_pool_size))?,
            edges_in_fusion_unit: f.field("edges_in_fusion_unit", Some(dual_defaults::edges_in_fusion_unit))?,
            enable_parallel_execution: f.field("enable_parallel_execution", Some(dual_defaults::enable_parallel_execution))?,
        };
        f.finish(true)?;
        Ok(out)
    }
}

struct_json!(PrimalModuleParallelConfig, self => { "thread_pool_size": self.thread_pool_size, "debug_sequential": self.debug_sequential, "prioritize_base_partition": self.prioritize_base_partition, "interleaving_base_fusion": self.interleaving_base_fusion, "pin_threads_to_cores": self.pin_threads_to_cores, "streaming_decode_mock_measure_interval": self.streaming_decode_mock_measure_interval, "streaming_decode_use_spin_lock": self.streaming_decode_use_spin_lock, "max_tree_size": self.max_tree_size, });

impl FromJson for PrimalModuleParallelConfig {
    fn from_json(value: &Value) -> Result<Self, Error> {
        let mut f = Fields::of(value, "PrimalModuleParallelConfig")?;
        let out = PrimalModuleParallelConfig {
            thread_pool_size: f.field("thread_pool_size", Some(primal_defaults::thread_pool_size))?,
            debug_sequential: f.field("debug_sequential", Some(primal_defaults::debug_sequential))?,
            prioritize_base_partition: f.field("prioritize_base_partition", Some(primal_defaults::prioritize_base_partition))?,
            interleaving_base_fusion: f.field("interleaving_base_fusion", Some(primal_defaults::interleaving_base_fusion))?,
            pin_threads_to_cores: f.field("pin_threads_to_cores", Some(primal_defaults::pin_threads_to_cores))?,
            streaming_decode_mock_measure_interval: f.optional("streaming_decode_mock_measure_interval")?,
            streaming_decode_use_spin_lock: f.field("streaming_decode_use_spin_lock", Some(primal_defaults::streaming_decode_use_spin_lock))?,
            max_tree_size: f.field("max_tree_size", Some(primal_defaults::max_tree_size))?,
        };
        f.finish(true)?;
        Ok(out)
    }
}

struct_json!(PrimalModuleParallelUnitEventTime, self => { "start": self.start, "end": self.end, "thread_index": self.thread_index });

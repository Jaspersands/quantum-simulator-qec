//! `serde`'s derives for MWPF's JSON types, as MWPF's Python wheels run them: `serde_json` built
//! with `arbitrary_precision` and without `preserve_order`, so an object's keys are visited in
//! sorted order, a number read straight from a value is reported as "number" and turns into an
//! unsigned integer only when it is written as one ("invalid number" otherwise), while one read
//! through a flattened struct's buffer is an integer or floating point; a struct reads from an
//! object or from an array of its fields; an error prints as `serde_json`'s does
//! (`Error("…", line: 0, column: 0)`).

use crate::mwpf::mwpf_solver::SolverSerialPluginsConfig;
use crate::mwpf::primal_module_serial::{primal_serial_default_configs, PrimalModuleSerialConfig};
use crate::mwpf::serde_json::{self, write_fields, FromJson, Number, ToJson, Value};
use crate::mwpf::util::*;
use crate::mwpf::json;
use crate::mwpf::num_traits::ToPrimitive;
use crate::mwpf::visualize::VisualizePosition;

/// `serde_json::Error` for a value: its `Debug` is `serde_json`'s.
pub struct JsonError(pub String);

impl std::fmt::Debug for JsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error({:?}, line: 0, column: 0)", self.0)
    }
}

impl std::fmt::Display for JsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// `serde_json::from_value`.
pub fn from_value<T: De>(value: Value) -> Result<T, JsonError> {
    T::de(&value).map_err(JsonError)
}

/// `serde_json::from_str`.
pub fn from_str<T: De>(text: &str) -> Result<T, JsonError> {
    let value: Value = serde_json::from_str(text).map_err(JsonError)?;
    from_value(value)
}

/// `serde::Deserialize` from a `serde_json::Value`.
pub trait De: Sized {
    fn de(value: &Value) -> Result<Self, String>;
}

/// `serde::de::Unexpected` for a value (an arbitrary-precision number is just "number").
fn unexpected(value: &Value) -> String {
    match value {
        Value::Null => "null".into(),
        Value::Bool(b) => format!("boolean `{b}`"),
        Value::Number(_) => "number".into(),
        Value::String(s) => format!("string {s:?}"),
        Value::Array(_) => "sequence".into(),
        Value::Object(_) => "map".into(),
    }
}

fn invalid_type(value: &Value, expected: &str) -> String {
    format!("invalid type: {}, expected {expected}", unexpected(value))
}

/// serde's `OneOf` list of field names.
fn unknown_field(key: &str, fields: &[&str]) -> String {
    match fields {
        [] => format!("unknown field `{key}`, there are no fields"),
        [a] => format!("unknown field `{key}`, expected `{a}`"),
        [a, b] => format!("unknown field `{key}`, expected `{a}` or `{b}`"),
        _ => format!("unknown field `{key}`, expected one of {}", fields.iter().map(|f| format!("`{f}`")).collect::<Vec<_>>().join(", ")),
    }
}

impl De for Value {
    fn de(value: &Value) -> Result<Self, String> {
        Ok(value.clone())
    }
}

impl De for f64 {
    fn de(value: &Value) -> Result<Self, String> {
        match value {
            Value::Number(Number::PosInt(u)) => Ok(*u as f64),
            Value::Number(Number::NegInt(i)) => Ok(*i as f64),
            Value::Number(Number::Float(x)) => Ok(*x),
            other => Err(invalid_type(other, "f64")),
        }
    }
}

impl De for usize {
    fn de(value: &Value) -> Result<Self, String> {
        match value {
            Value::Number(Number::PosInt(u)) => usize::try_from(*u).map_err(|_| "invalid number".to_string()),
            Value::Number(_) => Err("invalid number".into()),
            other => Err(invalid_type(other, "usize")),
        }
    }
}

impl De for bool {
    fn de(value: &Value) -> Result<Self, String> {
        match value {
            Value::Bool(b) => Ok(*b),
            other => Err(invalid_type(other, "a boolean")),
        }
    }
}

impl De for Rational {
    fn de(value: &Value) -> Result<Self, String> {
        f64::de(value).map(Rational::from)
    }
}

impl<T: De> De for Vec<T> {
    fn de(value: &Value) -> Result<Self, String> {
        match value {
            Value::Array(items) => items.iter().map(T::de).collect(),
            other => Err(invalid_type(other, "a sequence")),
        }
    }
}

impl<T: De> De for Option<T> {
    fn de(value: &Value) -> Result<Self, String> {
        match value {
            Value::Null => Ok(None),
            other => T::de(other).map(Some),
        }
    }
}

/// A tuple's elements, then the array's length checked as `serde_json` checks it.
fn tuple_items(value: &Value, size: usize) -> Result<&[Value], String> {
    let expected = format!("a tuple of size {size}");
    match value {
        Value::Array(items) => {
            if items.len() < size {
                return Err(format!("invalid length {}, expected {expected}", items.len()));
            }
            Ok(items)
        }
        other => Err(invalid_type(other, &expected)),
    }
}

fn no_more(items: &[Value], size: usize) -> Result<(), String> {
    if items.len() > size {
        return Err(format!("invalid length {}, expected fewer elements in array", items.len()));
    }
    Ok(())
}

impl<A: De, B: De> De for (A, B) {
    fn de(value: &Value) -> Result<Self, String> {
        let items = tuple_items(value, 2)?;
        let result = (A::de(&items[0])?, B::de(&items[1])?);
        no_more(items, 2)?;
        Ok(result)
    }
}

impl<A: De, B: De, C: De> De for (A, B, C) {
    fn de(value: &Value) -> Result<Self, String> {
        let items = tuple_items(value, 3)?;
        let result = (A::de(&items[0])?, B::de(&items[1])?, C::de(&items[2])?);
        no_more(items, 3)?;
        Ok(result)
    }
}

/// How a struct's field is filled when absent.
#[derive(Clone, Copy, PartialEq)]
enum Absent {
    /// an error
    Required,
    /// `#[serde(default)]`
    Default,
    /// an `Option`, `None` when absent from an object (but required in an array)
    Optional,
}

/// A derived struct's `visit_map` and `visit_seq`: calls `visit(field, value)` in the order serde
/// visits fields, and returns which were present.
fn visit_struct(
    value: &Value,
    name: &str,
    fields: &[&str],
    absent: &[Absent],
    deny_unknown_fields: bool,
    mut visit: impl FnMut(usize, &Value) -> Result<(), String>,
) -> Result<Vec<bool>, String> {
    let n = fields.len();
    let mut seen = vec![false; n];
    match value {
        Value::Object(map) => {
            for (key, x) in map.iter() {
                match fields.iter().position(|f| f == key) {
                    Some(k) => {
                        visit(k, x)?;
                        seen[k] = true;
                    }
                    None if deny_unknown_fields => return Err(unknown_field(key, fields)),
                    None => {}
                }
            }
            for k in 0..n {
                if !seen[k] && absent[k] == Absent::Required {
                    return Err(format!("missing field `{}`", fields[k]));
                }
            }
        }
        Value::Array(items) => {
            for k in 0..n {
                match items.get(k) {
                    Some(x) => {
                        visit(k, x)?;
                        seen[k] = true;
                    }
                    None if absent[k] == Absent::Default => {}
                    None => return Err(format!("invalid length {k}, expected struct {name} with {n} elements")),
                }
            }
            no_more(items, n)?;
        }
        other => return Err(invalid_type(other, &format!("struct {name}"))),
    }
    Ok(seen)
}

impl De for HyperEdge {
    fn de(value: &Value) -> Result<Self, String> {
        let (mut vertices, mut weight) = (None, None);
        visit_struct(value, "HyperEdge", &["vertices", "weight"], &[Absent::Required; 2], false, |k, x| {
            match k {
                0 => vertices = Some(De::de(x)?),
                _ => weight = Some(De::de(x)?),
            }
            Ok(())
        })?;
        Ok(HyperEdge::new(vertices.unwrap(), weight.unwrap()))
    }
}

impl De for SolverInitializer {
    fn de(value: &Value) -> Result<Self, String> {
        let (mut vertex_num, mut weighted_edges, mut heralds) = (None, None, None);
        visit_struct(value, "SolverInitializer", &["vertex_num", "weighted_edges", "heralds"], &[Absent::Required; 3], false, |k, x| {
            match k {
                0 => vertex_num = Some(De::de(x)?),
                1 => weighted_edges = Some(De::de(x)?),
                _ => heralds = Some(De::de(x)?),
            }
            Ok(())
        })?;
        Ok(SolverInitializer { vertex_num: vertex_num.unwrap(), weighted_edges: weighted_edges.unwrap(), heralds: heralds.unwrap() })
    }
}

impl De for SyndromePattern {
    fn de(value: &Value) -> Result<Self, String> {
        let (mut defect_vertices, mut erasures, mut heralds, mut override_weights) = (None, None, None, None);
        let absent = [Absent::Required, Absent::Required, Absent::Required, Absent::Optional];
        visit_struct(value, "SyndromePattern", &["defect_vertices", "erasures", "heralds", "override_weights"], &absent, false, |k, x| {
            match k {
                0 => defect_vertices = Some(De::de(x)?),
                1 => erasures = Some(De::de(x)?),
                2 => heralds = Some(De::de(x)?),
                _ => override_weights = Some(De::de(x)?),
            }
            Ok(())
        })?;
        Ok(SyndromePattern {
            defect_vertices: defect_vertices.unwrap(),
            erasures: erasures.unwrap(),
            heralds: heralds.unwrap(),
            override_weights: override_weights.unwrap_or(None),
        })
    }
}

impl De for VisualizePosition {
    fn de(value: &Value) -> Result<Self, String> {
        let mut ijt = [None; 3];
        visit_struct(value, "VisualizePosition", &["i", "j", "t"], &[Absent::Required; 3], false, |k, x| {
            ijt[k] = Some(f64::de(x)?);
            Ok(())
        })?;
        Ok(VisualizePosition::new(ijt[0].unwrap(), ijt[1].unwrap(), ijt[2].unwrap()))
    }
}

const PRIMAL_FIELDS: [&str; 3] = ["timeout", "cluster_node_limit", "only_solve_primal_once"];

impl De for PrimalModuleSerialConfig {
    fn de(value: &Value) -> Result<Self, String> {
        let mut config = PrimalModuleSerialConfig {
            timeout: primal_serial_default_configs::timeout(),
            cluster_node_limit: primal_serial_default_configs::cluster_node_limit(),
            only_solve_primal_once: primal_serial_default_configs::only_solve_primal_once(),
        };
        visit_struct(value, "PrimalModuleSerialConfig", &PRIMAL_FIELDS, &[Absent::Default; 3], true, |k, x| {
            match k {
                0 => config.timeout = De::de(x)?,
                1 => config.cluster_node_limit = De::de(x)?,
                _ => config.only_solve_primal_once = De::de(x)?,
            }
            Ok(())
        })?;
        Ok(config)
    }
}

/// `serde::de::Unexpected` for a value buffered as `serde`'s `Content`, where numbers are
/// integers or floating points.
fn content_unexpected(value: &Value) -> String {
    match value {
        Value::Number(Number::PosInt(u)) => format!("integer `{u}`"),
        Value::Number(Number::NegInt(i)) => format!("integer `{i}`"),
        Value::Number(Number::Float(x)) => {
            let text = format!("{x}");
            if x.is_finite() && !text.contains('.') {
                format!("floating point `{text}.0`")
            } else {
                format!("floating point `{text}`")
            }
        }
        other => unexpected(other),
    }
}

fn content_invalid_type(value: &Value, expected: &str) -> String {
    format!("invalid type: {}, expected {expected}", content_unexpected(value))
}

/// `SolverSerialPluginsConfig` (`#[serde(deny_unknown_fields)]`, its `PrimalModuleSerialConfig`
/// flattened beside the legacy `primal`): `primal` is read where its key falls, the other keys
/// are buffered and the flattened fields taken from the buffer in order, then any key left over
/// is unknown.
impl De for SolverSerialPluginsConfig {
    fn de(value: &Value) -> Result<Self, String> {
        let Value::Object(map) = value else {
            return Err(invalid_type(value, "struct SolverSerialPluginsConfig"));
        };
        let mut primal = None;
        let mut buffered = Vec::new();
        for (key, x) in map.iter() {
            if key == "primal" {
                primal = Option::<PrimalModuleSerialConfig>::de(x)?;
            } else {
                buffered.push((key, x));
            }
        }
        let mut flatten_primal = PrimalModuleSerialConfig::de(&json!({}))?;
        let mut left = Vec::new();
        for (key, x) in buffered {
            match key.as_str() {
                "timeout" => {
                    flatten_primal.timeout = match x {
                        Value::Number(Number::PosInt(u)) => *u as f64,
                        Value::Number(Number::NegInt(i)) => *i as f64,
                        Value::Number(Number::Float(f)) => *f,
                        other => return Err(content_invalid_type(other, "f64")),
                    }
                }
                "cluster_node_limit" => {
                    flatten_primal.cluster_node_limit = match x {
                        Value::Number(Number::PosInt(u)) => *u as usize,
                        Value::Number(Number::NegInt(i)) => return Err(format!("invalid value: integer `{i}`, expected usize")),
                        other => return Err(content_invalid_type(other, "usize")),
                    }
                }
                "only_solve_primal_once" => {
                    flatten_primal.only_solve_primal_once = match x {
                        Value::Bool(b) => *b,
                        other => return Err(content_invalid_type(other, "a boolean")),
                    }
                }
                _ => left.push(key),
            }
        }
        if let Some(key) = left.first() {
            return Err(format!("unknown field `{key}`"));
        }
        Ok(SolverSerialPluginsConfig { flatten_primal, primal })
    }
}

/// `serde_json::from_value` for the reading helpers, which the generic `FromJson` sites use.
macro_rules! from_json_via_de {
    ($($t:ty),*) => {$(
        impl FromJson for $t {
            fn from_json(value: &Value) -> Result<Self, serde_json::Error> {
                <$t as De>::de(value)
            }
        }
    )*};
}
from_json_via_de!(HyperEdge, SolverInitializer, SyndromePattern, VisualizePosition, PrimalModuleSerialConfig, SolverSerialPluginsConfig);

impl ToJson for Rational {
    fn to_json(&self) -> Value {
        self.to_f64().unwrap().to_json()
    }
}

impl ToJson for HyperEdge {
    fn to_json(&self) -> Value {
        json!({"vertices": self.vertices, "weight": self.weight})
    }
    fn write_text(&self, out: &mut String) {
        write_fields(&[("vertices", &self.vertices), ("weight", &self.weight)], out);
    }
}

impl ToJson for SolverInitializer {
    fn to_json(&self) -> Value {
        json!({"vertex_num": self.vertex_num, "weighted_edges": self.weighted_edges, "heralds": self.heralds})
    }
    fn write_text(&self, out: &mut String) {
        write_fields(&[("vertex_num", &self.vertex_num), ("weighted_edges", &self.weighted_edges), ("heralds", &self.heralds)], out);
    }
}

impl ToJson for SyndromePattern {
    fn to_json(&self) -> Value {
        json!({
            "defect_vertices": self.defect_vertices,
            "erasures": self.erasures,
            "heralds": self.heralds,
            "override_weights": self.override_weights,
        })
    }
    fn write_text(&self, out: &mut String) {
        write_fields(
            &[
                ("defect_vertices", &self.defect_vertices),
                ("erasures", &self.erasures),
                ("heralds", &self.heralds),
                ("override_weights", &self.override_weights),
            ],
            out,
        );
    }
}

impl ToJson for VisualizePosition {
    fn to_json(&self) -> Value {
        json!({"i": self.i, "j": self.j, "t": self.t})
    }
    fn write_text(&self, out: &mut String) {
        write_fields(&[("i", &self.i), ("j", &self.j), ("t", &self.t)], out);
    }
}

impl ToJson for PrimalModuleSerialConfig {
    fn to_json(&self) -> Value {
        json!({
            "timeout": self.timeout,
            "cluster_node_limit": self.cluster_node_limit,
            "only_solve_primal_once": self.only_solve_primal_once,
        })
    }
    fn write_text(&self, out: &mut String) {
        write_fields(
            &[("timeout", &self.timeout), ("cluster_node_limit", &self.cluster_node_limit), ("only_solve_primal_once", &self.only_solve_primal_once)],
            out,
        );
    }
}

impl ToJson for SolverSerialPluginsConfig {
    fn to_json(&self) -> Value {
        let mut value = self.flatten_primal.to_json();
        value["primal"] = self.primal.to_json();
        value
    }
    fn write_text(&self, out: &mut String) {
        let p = &self.flatten_primal;
        write_fields(
            &[
                ("timeout", &p.timeout),
                ("cluster_node_limit", &p.cluster_node_limit),
                ("only_solve_primal_once", &p.only_solve_primal_once),
                ("primal", &self.primal),
            ],
            out,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err<T: De + std::fmt::Debug>(value: Value) -> String {
        format!("{:?}", from_value::<T>(value).unwrap_err())
    }

    #[test]
    fn errors_are_serde_jsons() {
        assert_eq!(err::<SolverInitializer>(json!({"vertex_num": 3})), "Error(\"missing field `weighted_edges`\", line: 0, column: 0)");
        assert_eq!(
            err::<SolverInitializer>(json!({"a": 1, "vertex_num": 3, "weighted_edges": [5], "heralds": "z"})),
            "Error(\"invalid type: string \\\"z\\\", expected a sequence\", line: 0, column: 0)"
        );
        assert_eq!(err::<SolverInitializer>(json!([3, []])), "Error(\"invalid length 2, expected struct SolverInitializer with 3 elements\", line: 0, column: 0)");
        assert_eq!(err::<SolverSerialPluginsConfig>(json!({"bogus": 1})), "Error(\"unknown field `bogus`\", line: 0, column: 0)");
        assert_eq!(err::<SolverSerialPluginsConfig>(json!({"bogus": 1, "timeout": "x"})), "Error(\"invalid type: string \\\"x\\\", expected f64\", line: 0, column: 0)");
        assert_eq!(err::<SolverSerialPluginsConfig>(json!({"cluster_node_limit": 1.5})), "Error(\"invalid type: floating point `1.5`, expected usize\", line: 0, column: 0)");
        assert_eq!(err::<SolverSerialPluginsConfig>(json!({"primal": {"cluster_node_limit": -1}})), "Error(\"invalid number\", line: 0, column: 0)");
        assert_eq!(err::<SolverSerialPluginsConfig>(json!({"primal": [1.0, 5, true, 4]})), "Error(\"invalid length 4, expected fewer elements in array\", line: 0, column: 0)");
        assert_eq!(
            err::<SolverSerialPluginsConfig>(json!({"primal": {"bogus": 1}})),
            "Error(\"unknown field `bogus`, expected one of `timeout`, `cluster_node_limit`, `only_solve_primal_once`\", line: 0, column: 0)"
        );
    }

    #[test]
    fn text_keeps_declared_order() {
        let initializer = SolverInitializer { vertex_num: 3, weighted_edges: vec![HyperEdge::new(vec![0, 1], Rational::from(1.5))], heralds: vec![] };
        assert_eq!(serde_json::to_string(&initializer).unwrap(), r#"{"vertex_num":3,"weighted_edges":[{"vertices":[0,1],"weight":1.5}],"heralds":[]}"#);
    }
}

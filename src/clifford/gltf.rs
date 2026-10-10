//! Stim's 3D diagrams as glTF (`timeline-3d`, `matchgraph-3d`), written as Stim writes them:
//! a port of `gltf.cc`, `json_obj.cc`, `basic_3d_diagram.cc`, `gate_data_3d.cc`,
//! `timeline_3d_drawer.cc` and `match_graph_3d_drawer.cc` (Stim, Apache-2.0; the gate texture
//! is Stim's own image). Arithmetic is in `f32` wherever Stim's is in `float`, so the vertex
//! buffers come out bit for bit the same.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use super::ir::{self, GateTarget, Instruction, Item};
use super::transform::final_qubit_coordinates;
use crate::dem_program::{fmt_g, DemInstr, DemProgram, OBS};
use crate::gate_data::{FLAG_TARGETS_COMBINERS, FLAG_TARGETS_PAIRS};

// ---------------------------------------------------------------------------------------------
// JSON, as Stim's `JsonObj::write` with no indentation: map keys sorted, floats at the stream's
// precision (6), doubles at 15 digits.

#[derive(Clone, Debug)]
pub enum Json {
    Map(BTreeMap<String, Json>),
    Arr(Vec<Json>),
    Bool(bool),
    F32(f32),
    F64(f64),
    Int(i64),
    UInt(u64),
    Str(String),
}

impl Json {
    fn map<const N: usize>(entries: [(&str, Json); N]) -> Json {
        Json::Map(entries.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
    }

    pub fn write(&self, out: &mut String) {
        match self {
            Json::Map(m) => {
                out.push('{');
                for (k, (key, v)) in m.iter().enumerate() {
                    if k > 0 {
                        out.push(',');
                    }
                    write_str(key, out);
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
            Json::Arr(a) => {
                out.push('[');
                for (k, v) in a.iter().enumerate() {
                    if k > 0 {
                        out.push(',');
                    }
                    v.write(out);
                }
                out.push(']');
            }
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::F32(v) => out.push_str(&num(*v as f64, 6)),
            Json::F64(v) => out.push_str(&num(*v, 15)),
            Json::Int(v) => {
                let _ = write!(out, "{v}");
            }
            Json::UInt(v) => {
                let _ = write!(out, "{v}");
            }
            Json::Str(s) => write_str(s, out),
        }
    }
}

/// C++'s `ostream << double` at a precision: `%g`, with `-0` for negative zero.
fn num(v: f64, precision: usize) -> String {
    if v == 0.0 && v.is_sign_negative() {
        return "-0".into();
    }
    fmt_g(v, precision)
}

fn write_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '\0' => out.push_str("\\0"),
            '\n' => out.push_str("\\n"),
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Stim's base64 (standard alphabet, `=` padding).
pub fn base64(data: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        out.push(A[(n >> 18) as usize & 63] as char);
        out.push(A[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { A[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { A[n as usize & 63] as char } else { '=' });
    }
    out
}

// ---------------------------------------------------------------------------------------------
// The scene: objects in an arena, numbered per kind in the order a depth-first visit first meets
// them (Stim numbers shared objects once, by pointer).

const GL_FLOAT: u64 = 5126;
const GL_ARRAY_BUFFER: u64 = 34962;
const GL_TRIANGLES: u64 = 4;
const GL_TRIANGLE_FAN: u64 = 6;
const GL_LINES: u64 = 1;
const GL_LINE_STRIP: u64 = 3;
const GL_LINE_LOOP: u64 = 2;
const GL_CLAMP_TO_EDGE: u64 = 33071;
const GL_NEAREST: u64 = 9728;

type C3 = [f32; 3];
type C2 = [f32; 2];

enum Obj {
    Buffer { name: String, dim: usize, data: Vec<f32> },
    Sampler,
    Image(String),
    Texture { sampler: usize, image: usize },
    Material { rgba: [f32; 4], metallic: f32, roughness: f32, double_sided: bool, texture: Option<usize> },
    Primitive { mode: u64, pos: usize, tex: Option<usize>, material: usize },
    Mesh(Vec<usize>),
    Node { mesh: usize, translation: C3 },
}

#[derive(Default)]
struct Arena {
    objs: Vec<Obj>,
}

impl Arena {
    fn add(&mut self, o: Obj) -> usize {
        self.objs.push(o);
        self.objs.len() - 1
    }

    fn buf3(&mut self, name: &str, v: &[C3]) -> usize {
        self.add(Obj::Buffer { name: name.into(), dim: 3, data: v.iter().flatten().copied().collect() })
    }

    fn buf2(&mut self, name: &str, v: &[C2]) -> usize {
        self.add(Obj::Buffer { name: name.into(), dim: 2, data: v.iter().flatten().copied().collect() })
    }

    fn material(&mut self, rgba: [f32; 4], metallic: f32, roughness: f32, double_sided: bool, texture: Option<usize>) -> usize {
        self.add(Obj::Material { rgba, metallic, roughness, double_sided, texture })
    }

    fn prim(&mut self, mode: u64, pos: usize, tex: Option<usize>, material: usize) -> usize {
        self.add(Obj::Primitive { mode, pos, tex, material })
    }

    /// The glTF document of a scene of these nodes (Stim's `GltfScene::to_json`).
    fn scene_json(&self, nodes: &[usize]) -> Json {
        // kind -> objects in first-visit order.
        let mut order: BTreeMap<&'static str, Vec<usize>> = BTreeMap::new();
        let mut index: BTreeMap<(&'static str, usize), usize> = BTreeMap::new();
        let mut see = |kind: &'static str, id: usize, order: &mut BTreeMap<&'static str, Vec<usize>>| {
            if !index.contains_key(&(kind, id)) {
                let list = order.entry(kind).or_default();
                index.insert((kind, id), list.len());
                list.push(id);
            }
        };
        order.entry("scenes").or_default().push(usize::MAX);
        for &n in nodes {
            self.visit(n, &mut |k, id| see(k, id, &mut order));
        }
        let idx = |kind: &'static str, id: usize| -> Json {
            let list = &order[kind];
            Json::UInt(list.iter().position(|&x| x == id).unwrap() as u64)
        };
        let mut result = BTreeMap::new();
        result.insert("scene".to_string(), Json::Int(0));
        result.insert("asset".to_string(), Json::map([("version", Json::Str("2.0".into()))]));
        for (&kind, ids) in &order {
            let items: Vec<Json> = ids
                .iter()
                .map(|&id| {
                    if kind == "scenes" {
                        return Json::map([("nodes", Json::Arr(nodes.iter().map(|&n| idx("nodes", n)).collect()))]);
                    }
                    self.item_json(kind, id, &idx)
                })
                .collect();
            result.insert(kind.to_string(), Json::Arr(items));
        }
        Json::Map(result)
    }

    fn visit(&self, id: usize, f: &mut dyn FnMut(&'static str, usize)) {
        match &self.objs[id] {
            Obj::Node { mesh, .. } => {
                f("nodes", id);
                self.visit(*mesh, f);
            }
            Obj::Mesh(prims) => {
                f("meshes", id);
                for &p in prims {
                    self.visit(p, f);
                }
            }
            Obj::Primitive { pos, tex, material, .. } => {
                self.visit(*pos, f);
                if let Some(t) = tex {
                    self.visit(*t, f);
                }
                self.visit(*material, f);
            }
            Obj::Buffer { .. } => {
                f("buffers", id);
                f("bufferViews", id);
                f("accessors", id);
            }
            Obj::Material { texture, .. } => {
                f("materials", id);
                if let Some(t) = texture {
                    self.visit(*t, f);
                }
            }
            Obj::Texture { sampler, image } => {
                f("textures", id);
                self.visit(*sampler, f);
                self.visit(*image, f);
            }
            Obj::Sampler => f("samplers", id),
            Obj::Image(_) => f("images", id),
        }
    }

    fn item_json(&self, kind: &str, id: usize, idx: &dyn Fn(&'static str, usize) -> Json) -> Json {
        let f32s = |v: &[f32]| Json::Arr(v.iter().map(|&x| Json::F32(x)).collect());
        match (&self.objs[id], kind) {
            (Obj::Node { mesh, translation }, _) => Json::map([("mesh", idx("meshes", *mesh)), ("translation", f32s(translation))]),
            (Obj::Mesh(prims), _) => Json::map([(
                "primitives",
                Json::Arr(
                    prims
                        .iter()
                        .map(|&p| {
                            let Obj::Primitive { mode, pos, tex, material } = &self.objs[p] else { unreachable!() };
                            let mut attrs = BTreeMap::new();
                            attrs.insert("POSITION".to_string(), idx("accessors", *pos));
                            if let Some(t) = tex {
                                attrs.insert("TEXCOORD_0".to_string(), idx("accessors", *t));
                            }
                            Json::map([("attributes", Json::Map(attrs)), ("material", idx("materials", *material)), ("mode", Json::UInt(*mode))])
                        })
                        .collect(),
                ),
            )]),
            (Obj::Buffer { name, dim, data }, kind) => {
                let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
                let n = data.len() / dim;
                match kind {
                    "buffers" => Json::map([("name", Json::Str(name.clone())), ("uri", Json::Str(format!("data:application/octet-stream;base64,{}", base64(&bytes)))), ("byteLength", Json::UInt(bytes.len() as u64))]),
                    "bufferViews" => Json::map([("name", Json::Str(name.clone())), ("buffer", idx("buffers", id)), ("byteOffset", Json::Int(0)), ("byteLength", Json::UInt(bytes.len() as u64)), ("target", Json::UInt(GL_ARRAY_BUFFER))]),
                    _ => {
                        let (mut lo, mut hi) = (vec![0f32; *dim], vec![0f32; *dim]);
                        if n > 0 {
                            lo = vec![f32::INFINITY; *dim];
                            hi = vec![f32::NEG_INFINITY; *dim];
                            for v in data.chunks(*dim) {
                                for k in 0..*dim {
                                    lo[k] = if v[k] < lo[k] { v[k] } else { lo[k] };
                                    hi[k] = if hi[k] < v[k] { v[k] } else { hi[k] };
                                }
                            }
                        }
                        let d = |v: &[f32]| Json::Arr(v.iter().map(|&x| Json::F64(x as f64)).collect());
                        Json::map([
                            ("name", Json::Str(name.clone())),
                            ("bufferView", idx("bufferViews", id)),
                            ("byteOffset", Json::Int(0)),
                            ("componentType", Json::UInt(GL_FLOAT)),
                            ("count", Json::UInt(n as u64)),
                            ("type", Json::Str(format!("VEC{dim}"))),
                            ("min", d(&lo)),
                            ("max", d(&hi)),
                        ])
                    }
                }
            }
            (Obj::Material { rgba, metallic, roughness, double_sided, texture }, _) => {
                let mut pbr = BTreeMap::new();
                pbr.insert("baseColorFactor".to_string(), f32s(rgba));
                pbr.insert("metallicFactor".to_string(), Json::F32(*metallic));
                pbr.insert("roughnessFactor".to_string(), Json::F32(*roughness));
                if let Some(t) = texture {
                    pbr.insert("baseColorTexture".to_string(), Json::map([("index", idx("textures", *t)), ("texCoord", Json::Int(0))]));
                }
                Json::map([("pbrMetallicRoughness", Json::Map(pbr)), ("doubleSided", Json::Bool(*double_sided))])
            }
            (Obj::Texture { .. }, _) => Json::map([("sampler", Json::Int(0)), ("source", Json::Int(0))]),
            (Obj::Sampler, _) => Json::map([("magFilter", Json::UInt(GL_NEAREST)), ("minFilter", Json::UInt(GL_NEAREST)), ("wrapS", Json::UInt(GL_CLAMP_TO_EDGE)), ("wrapT", Json::UInt(GL_CLAMP_TO_EDGE))]),
            (Obj::Image(uri), _) => Json::map([("uri", Json::Str(uri.clone()))]),
            (Obj::Primitive { .. }, _) => unreachable!(),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Gate meshes (Stim's `make_gate_primitives`).

const CONTROL_RADIUS: f32 = 0.4;
const TEXTURE_PNG_BASE64: &str = include_str!("gltf_texture.b64");

fn circle_loop(a: &mut Arena, n: usize, r: f32, repeat_boundary: bool) -> usize {
    let mut v: Vec<C3> = vec![[0.0, r, 0.0]];
    for k in 1..n {
        let t = k as f32 * 3.141_592_653_59_f32 * 2.0 / n as f32;
        v.push([0.0, t.cos() * r, t.sin() * r]);
    }
    if repeat_boundary {
        v.push([0.0, r, 0.0]);
    }
    a.buf3("circle_loop", &v)
}

fn black(a: &mut Arena) -> usize {
    a.material([0.0, 0.0, 0.0, 1.0], 1.0, 1.0, true, None)
}

fn white(a: &mut Arena) -> usize {
    a.material([1.0, 1.0, 1.0, 1.0], 0.4, 0.5, true, None)
}

fn gate_meshes(a: &mut Arena) -> BTreeMap<String, usize> {
    let mut out: BTreeMap<String, usize> = BTreeMap::new();
    // The cube is a square (Stim's `actually_just_square`).
    let (v000, v001, v010, v011) = ([0.0, 0.5, 0.5], [0.0, 0.5, -0.5], [0.0, -0.5, 0.5], [0.0, -0.5, -0.5]);
    let cube = a.buf3("cube", &[v000, v001, v010, v001, v011, v010, v011, v001, v010, v010, v001, v000]);
    let image = a.add(Obj::Image(format!("data:image/png;base64,{TEXTURE_PNG_BASE64}")));
    let sampler = a.add(Obj::Sampler);
    let texture = a.add(Obj::Texture { sampler, image });
    let material = a.material([1.0, 1.0, 1.0, 1.0], 0.4, 0.5, false, Some(texture));
    let tiles: &[(&str, usize, usize)] = &[
        ("X", 0, 6), ("Y", 0, 7), ("Z", 0, 8), ("H_YZ", 1, 6), ("H", 1, 7), ("H_XY", 1, 8),
        ("SQRT_X", 2, 6), ("SQRT_Y", 2, 7), ("S", 2, 8), ("SQRT_X_DAG", 3, 6), ("SQRT_Y_DAG", 3, 7), ("S_DAG", 3, 8),
        ("MX", 4, 6), ("MY", 4, 7), ("M", 4, 8), ("RX", 5, 6), ("RY", 5, 7), ("R", 5, 8),
        ("MRX", 6, 6), ("MRY", 6, 7), ("MR", 6, 8), ("X_ERROR", 7, 6), ("Y_ERROR", 7, 7), ("Z_ERROR", 7, 8),
        ("E:X", 8, 6), ("E:Y", 8, 7), ("E:Z", 8, 8),
        ("ELSE_CORRELATED_ERROR:X", 9, 6), ("ELSE_CORRELATED_ERROR:Y", 9, 7), ("ELSE_CORRELATED_ERROR:Z", 9, 8),
        ("MPP:X", 10, 6), ("MPP:Y", 10, 7), ("MPP:Z", 10, 8), ("SQRT_XX", 11, 6), ("SQRT_YY", 11, 7), ("SQRT_ZZ", 11, 8),
        ("SQRT_XX_DAG", 12, 6), ("SQRT_YY_DAG", 12, 7), ("SQRT_ZZ_DAG", 12, 8),
        ("X:REC", 13, 6), ("Y:REC", 13, 7), ("Z:REC", 13, 8), ("X:SWEEP", 14, 6), ("Y:SWEEP", 14, 7), ("Z:SWEEP", 14, 8),
        ("I", 0, 6), ("C_XYZ", 1, 9), ("C_NXYZ", 6, 10), ("C_XNYZ", 7, 10), ("C_XYNZ", 8, 10), ("C_ZYX", 2, 9),
        ("C_NZYX", 9, 10), ("C_ZNYX", 10, 10), ("C_ZYNX", 11, 10), ("H_NXY", 12, 10), ("H_NXZ", 13, 10), ("H_NYZ", 14, 10),
        ("II", 15, 10), ("II_ERROR", 15, 11), ("I_ERROR", 15, 8), ("DEPOLARIZE1", 3, 9), ("DEPOLARIZE2", 4, 9),
        ("ISWAP", 5, 9), ("ISWAP_DAG", 6, 9), ("SWAP", 7, 9), ("PAULI_CHANNEL_1", 8, 9), ("PAULI_CHANNEL_2", 9, 9),
        ("MXX", 10, 9), ("MYY", 11, 9), ("MZZ", 12, 9), ("MPAD", 13, 9), ("HERALDED_ERASE", 14, 9),
        ("HERALDED_PAULI_CHANNEL_1", 15, 9), ("SPP:X", 0, 10), ("SPP:Y", 1, 10), ("SPP:Z", 2, 10),
        ("SPP_DAG:X", 3, 10), ("SPP_DAG:Y", 4, 10), ("SPP_DAG:Z", 5, 10),
    ];
    let d: f32 = 1.0 / 16.0;
    for &(name, x, y) in tiles {
        let (dx, dy) = (d * x as f32, d * y as f32);
        let (v00, v01, v10, v11) = ([dx, dy], [dx, dy + d], [dx + d, dy], [dx + d, dy + d]);
        let tex = a.buf2(&format!("tex_coords_gate_{name}"), &[v10, v00, v11, v00, v01, v11, v11, v10, v01, v01, v10, v00]);
        let prim = a.prim(GL_TRIANGLES, cube, Some(tex), material);
        let mesh = a.add(Obj::Mesh(vec![prim]));
        out.entry(name.to_string()).or_insert(mesh);
    }
    // X control, and the two swap controls.
    let h = CONTROL_RADIUS * 2f32.sqrt() * 0.8;
    for (key, cross) in [("X_CONTROL", vec![[0.0, -CONTROL_RADIUS, 0.0], [0.0, CONTROL_RADIUS, 0.0], [0.0, 0.0, -CONTROL_RADIUS], [0.0, 0.0, CONTROL_RADIUS]]), ("XSWAP", vec![[0.0, -h, -h], [0.0, h, h], [0.0, -h, h], [0.0, h, -h]])] {
        let line_cross = a.buf3(if key == "X_CONTROL" { "control_x_line_cross" } else { "control_xswap_line_cross" }, &cross);
        let circle = circle_loop(a, 16, CONTROL_RADIUS, true);
        let (bl, wh) = (black(a), white(a));
        let p1 = a.prim(GL_TRIANGLE_FAN, circle, None, wh);
        let p2 = a.prim(GL_LINE_STRIP, circle, None, bl);
        let p3 = a.prim(GL_LINES, line_cross, None, bl);
        let mesh = a.add(Obj::Mesh(vec![p1, p2, p3]));
        out.entry(key.to_string()).or_insert(mesh);
    }
    {
        let line_cross = a.buf3("control_zswap_line_cross", &[[0.0, -h, -h], [0.0, h, h], [0.0, -h, h], [0.0, h, -h]]);
        let circle = circle_loop(a, 16, CONTROL_RADIUS, true);
        let (bl, wh) = (black(a), white(a));
        let p1 = a.prim(GL_TRIANGLE_FAN, circle, None, bl);
        let p2 = a.prim(GL_LINES, line_cross, None, wh);
        let mesh = a.add(Obj::Mesh(vec![p1, p2]));
        out.entry("ZSWAP".to_string()).or_insert(mesh);
    }
    {
        let gray = a.material([0.5, 0.5, 0.5, 1.0], 1.0, 1.0, true, None);
        let bl = black(a);
        let tri = circle_loop(a, 3, CONTROL_RADIUS, false);
        let p1 = a.prim(GL_LINE_LOOP, tri, None, bl);
        let p2 = a.prim(GL_TRIANGLES, tri, None, gray);
        let mesh = a.add(Obj::Mesh(vec![p1, p2]));
        out.entry("Y_CONTROL".to_string()).or_insert(mesh);
    }
    {
        let circle = circle_loop(a, 16, CONTROL_RADIUS, true);
        let bl = black(a);
        let p = a.prim(GL_TRIANGLE_FAN, circle, None, bl);
        let mesh = a.add(Obj::Mesh(vec![p]));
        out.entry("Z_CONTROL".to_string()).or_insert(mesh);
    }
    for excited in [false, true] {
        let c1 = circle_loop(a, 8, CONTROL_RADIUS, true);
        let c2 = circle_loop(a, 8, CONTROL_RADIUS, true);
        let c3 = circle_loop(a, 8, CONTROL_RADIUS, true);
        for (id, rot) in [(c2, 0), (c3, 1)] {
            if let Obj::Buffer { data, .. } = &mut a.objs[id] {
                for e in data.chunks_mut(3) {
                    if rot == 0 {
                        e.swap(1, 2);
                        e.swap(0, 1);
                    } else {
                        e.swap(0, 1);
                        e.swap(1, 2);
                    }
                }
            }
        }
        let m = a.material(if excited { [1.0, 0.5, 0.5, 1.0] } else { [0.0, 0.0, 0.0, 1.0] }, 1.0, 1.0, true, None);
        let ps: Vec<usize> = [c1, c2, c3].iter().map(|&c| a.prim(GL_TRIANGLE_FAN, c, None, m)).collect();
        let mesh = a.add(Obj::Mesh(ps));
        out.entry(if excited { "EXCITED_DETECTOR" } else { "DETECTOR" }.to_string()).or_insert(mesh);
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Stim's `Basic3dDiagram`.

#[derive(Default)]
pub struct Basic3d {
    elements: Vec<(String, C3)>,
    lines: Vec<C3>,
    red: Vec<C3>,
    blue: Vec<C3>,
    purple: Vec<C3>,
}

impl Basic3d {
    /// The glTF document (Stim's `to_gltf_scene().to_json()` written out).
    pub fn to_gltf(&self) -> Result<String, String> {
        let mut a = Arena::default();
        let mut nodes = Vec::new();
        let black_m = black(&mut a);
        let red_m = a.material([1.0, 0.0, 0.0, 1.0], 1.0, 1.0, true, None);
        let blue_m = a.material([0.0, 0.0, 1.0, 1.0], 1.0, 1.0, true, None);
        let purple_m = a.material([1.0, 0.0, 1.0, 1.0], 1.0, 1.0, true, None);
        let bufs = [a.buf3("buf_scattered_lines", &self.lines), a.buf3("buf_red_scattered_lines", &self.red), a.buf3("buf_blue_scattered_lines", &self.blue), a.buf3("buf_purple_scattered_lines", &self.purple)];
        let gates = gate_meshes(&mut a);
        for (piece, center) in &self.elements {
            let mesh = *gates.get(piece).ok_or_else(|| format!("Basic3dDiagram unknown gate piece: {piece}"))?;
            nodes.push(a.add(Obj::Node { mesh, translation: *center }));
        }
        for ((buf, m), list) in bufs.iter().zip([black_m, red_m, blue_m, purple_m]).zip([&self.lines, &self.red, &self.blue, &self.purple]) {
            if !list.is_empty() {
                let p = a.prim(GL_LINES, *buf, None, m);
                let mesh = a.add(Obj::Mesh(vec![p]));
                nodes.push(a.add(Obj::Node { mesh, translation: [0.0; 3] }));
            }
        }
        if nodes.is_empty() {
            #[rustfmt::skip]
            let msg: Vec<C3> = [
                (0.0, 0.0), (0.0, 2.0), (0.0, 2.0), (1.0, 2.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0), (1.0, 0.0),
                (2.0, 1.0), (3.0, 1.0), (2.0, 0.0), (2.0, 1.0), (2.5, 0.0), (2.5, 1.0), (3.0, 0.0), (3.0, 1.0),
                (4.0, 1.0), (4.0, -1.0), (4.0, 1.0), (5.0, 1.0), (5.0, 1.0), (5.0, 0.0), (4.0, 0.0), (5.0, 0.0),
                (6.0, 0.0), (6.0, 2.0), (5.5, 1.5), (6.5, 1.5),
                (7.0, -1.0), (8.0, 1.0), (7.0, 1.0), (7.5, 0.0),
            ].iter().map(|&(x, y)| [x, y, 0.0]).collect();
            let buf = a.buf3("buf_blue_scattered_lines", &msg);
            let p = a.prim(GL_LINES, buf, None, red_m);
            let mesh = a.add(Obj::Mesh(vec![p]));
            nodes.push(a.add(Obj::Node { mesh, translation: [0.0; 3] }));
        }
        let mut out = String::new();
        a.scene_json(&nodes).write(&mut out);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------------
// The circuit timeline in 3D (Stim's `DiagramTimeline3DDrawer`).

fn flattened_2d(c: &[f64]) -> C2 {
    let mut x = c.first().map_or(0.0, |&v| v as f32);
    let mut y = c.get(1).map_or(0.0, |&v| v as f32);
    for (k, &v) in c.iter().enumerate().skip(2) {
        x += v as f32 / k as f32;
        y += v as f32 / (k * k) as f32;
    }
    [x, y]
}

fn lt2(a: C2, b: C2) -> bool {
    for k in 0..2 {
        if a[k] != b[k] {
            return a[k] < b[k];
        }
    }
    false
}

fn min_max2(v: &[C2]) -> (C2, C2) {
    if v.is_empty() {
        return ([0.0; 2], [0.0; 2]);
    }
    let (mut lo, mut hi) = ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]);
    for c in v {
        for k in 0..2 {
            lo[k] = if c[k] < lo[k] { c[k] } else { lo[k] };
            hi[k] = if hi[k] < c[k] { c[k] } else { hi[k] };
        }
    }
    (lo, hi)
}

/// Stim's `pick_coords_for_circuit`: each qubit's 2D position and the used qubits' bounds.
fn pick_coords(c: &ir::Circuit, nq: usize, used: &BTreeSet<u32>) -> (Vec<C2>, (C2, C2)) {
    let coordinates = final_qubit_coordinates(c);
    // FlattenedCoords::from(set, 1) for a set of these coordinates and no slices.
    let mut coords: Vec<C2> = (0..nq).map(|q| match coordinates.get(&(q as u64)) {
        Some(v) if !v.is_empty() => flattened_2d(v),
        _ => [q as f32, 0.0],
    }).collect();
    let set_used: Vec<u64> = coordinates.keys().copied().collect();
    let characteristic = if set_used.is_empty() {
        1.0
    } else {
        let mut biggest = [f32::NEG_INFINITY; 2];
        for &q in &set_used {
            let p = coords[q as usize];
            if lt2(biggest, p) {
                biggest = p;
            }
        }
        let mut closest = f32::INFINITY;
        for p in &coords {
            if biggest == *p {
                continue;
            }
            let d = [biggest[0] - p[0], biggest[1] - p[1]];
            let d2 = d[0] * d[0] + d[1] * d[1];
            if d2 < closest {
                closest = d2;
            }
        }
        let r = closest.sqrt();
        if r == f32::INFINITY {
            1.0
        } else {
            r
        }
    };
    let scale = 1.0f32 / characteristic;
    for p in coords.iter_mut() {
        p[0] *= scale;
        p[1] *= scale;
    }
    if !set_used.is_empty() {
        let used_coords: Vec<C2> = set_used.iter().map(|&q| coords[q as usize]).collect();
        let (lo, _) = min_max2(&used_coords);
        let offset = [lo[0] * -1.0 + 16.0, lo[1] * -1.0 + 16.0];
        for p in coords.iter_mut() {
            p[0] += offset[0];
            p[1] += offset[1];
        }
    }
    let mut default_y = 0f32;
    for &q in coordinates.keys() {
        let v = coords[q as usize][1] - 1.0;
        default_y = if v < default_y { v } else { default_y };
    }
    for (q, p) in coords.iter_mut().enumerate() {
        if !coordinates.contains_key(&(q as u64)) {
            *p = [q as f32, default_y];
        }
    }
    let mut used_coords: Vec<C2> = used.iter().map(|&q| coords[q as usize]).collect();
    if used_coords.is_empty() {
        used_coords.push([0.0, 0.0]);
    }
    let bounds = min_max2(&used_coords);
    (coords, bounds)
}

fn trans(m: usize, xy: C2) -> C3 {
    [-(m as f32), xy[0] * -2.0, xy[1] * -2.0]
}

fn add_used_qubits(c: &ir::Circuit, out: &mut BTreeSet<u32>) {
    for it in &c.items {
        match it {
            Item::Op(op) => {
                for t in &op.targets {
                    if t.is_pauli() || t.is_qubit() {
                        out.insert(t.value());
                    }
                }
            }
            Item::Repeat { body, .. } => add_used_qubits(body, out),
        }
    }
}

fn two_qubit_pieces(name: &'static str) -> (&'static str, &'static str) {
    match name {
        "CX" => ("Z", "X"),
        "CY" => ("Z", "Y"),
        "CZ" => ("Z", "Z"),
        "XCX" => ("X", "X"),
        "XCY" => ("X", "Y"),
        "XCZ" => ("X", "Z"),
        "YCX" => ("Y", "X"),
        "YCY" => ("Y", "Y"),
        "YCZ" => ("Y", "Z"),
        "CXSWAP" => ("ZSWAP", "XSWAP"),
        "CZSWAP" => ("ZSWAP", "ZSWAP"),
        "SWAPCX" => ("XSWAP", "ZSWAP"),
        n => (n, n),
    }
}

struct Timeline3d {
    out: Basic3d,
    coords: Vec<C2>,
    bounds: (C2, C2),
    cur_moment: usize,
    tick_start_moment: usize,
    used: Vec<bool>,
    has_ticks: bool,
    loop_starts: Vec<usize>,
    nesting: usize,
}

impl Timeline3d {
    fn mq(&self, m: usize, q: u32) -> C3 {
        trans(m, self.coords[q as usize])
    }

    fn start_next_moment(&mut self) {
        self.cur_moment += 1;
        self.used.iter_mut().for_each(|u| *u = false);
    }

    fn reserve(&mut self, targets: &[GateTarget]) {
        let qs: Vec<usize> = targets.iter().filter(|t| t.is_pauli() || t.is_qubit()).map(|t| t.value() as usize).collect();
        if qs.iter().any(|&q| self.used[q]) {
            self.start_next_moment();
        }
        for q in qs {
            self.used[q] = true;
        }
    }

    fn tick(&mut self) {
        if self.has_ticks && self.cur_moment > self.tick_start_moment {
            let (lo, hi) = self.bounds;
            let (x1, x2) = (lo[0] - 0.2, lo[0] - 0.4);
            let (y1, y2) = (lo[1] - 0.25, hi[1] + 0.25);
            let s = self.tick_start_moment;
            let m = self.cur_moment;
            let mut p = [trans(s, [x1, y1]), trans(s, [x1, y2]), trans(s, [x2, y1]), trans(s, [x2, y2]), trans(m, [x1, y1]), trans(m, [x1, y2]), trans(m, [x2, y1]), trans(m, [x2, y2])];
            for (k, c) in p.iter_mut().enumerate() {
                c[0] = (c[0] as f64 + if k < 4 { 0.25 } else { -0.25 }) as f32;
            }
            for k in [0, 2, 1, 3, 2, 3, 2, 6, 3, 7, 4, 6, 5, 7, 6, 7] {
                self.out.blue.push(p[k]);
            }
        }
        self.start_next_moment();
        self.tick_start_moment = self.cur_moment;
    }

    fn line(&mut self, a: C3, b: C3) {
        self.out.lines.push(a);
        let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if norm as f64 > 2.2 {
            let mut c = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5];
            c[0] = (c[0] as f64 - 0.25) as f32;
            self.out.lines.push(c);
            self.out.lines.push(c);
        }
        self.out.lines.push(b);
    }

    fn end_point(&mut self, center: C3, piece: &str) {
        let key = match piece {
            "X" => "X_CONTROL",
            "Y" => "Y_CONTROL",
            "Z" => "Z_CONTROL",
            p => p,
        };
        self.out.elements.push((key.to_string(), center));
    }

    fn feedback(&mut self, gate: &str, q: GateTarget, bit: GateTarget) {
        let mut key = gate.to_string();
        if bit.is_sweep() {
            key.push_str(":SWEEP");
        } else if bit.is_record() {
            key.push_str(":REC");
        }
        let c = self.mq(self.cur_moment, q.value());
        self.out.elements.push((key, c));
    }

    fn two_qubit(&mut self, name: &'static str, t: &[GateTarget]) {
        self.reserve(t);
        let (a, b) = (t[0], t[1]);
        let ends = two_qubit_pieces(name);
        if a.is_record() || a.is_sweep() {
            self.feedback(ends.1, b, a);
            return;
        }
        if b.is_record() || b.is_sweep() {
            self.feedback(ends.0, a, b);
            return;
        }
        let (c1, c2) = (self.mq(self.cur_moment, a.value()), self.mq(self.cur_moment, b.value()));
        self.end_point(c1, ends.0);
        self.end_point(c2, ends.1);
        self.line(c1, c2);
    }

    fn pauli_product(&mut self, name: &str, t: &[GateTarget]) {
        self.reserve(t);
        let mut prev: Option<C3> = None;
        for g in t.iter().filter(|g| !g.is_combiner()) {
            let mut key = name.to_string();
            if g.is_x() {
                key.push_str(":X");
            } else if g.is_y() {
                key.push_str(":Y");
            } else if g.is_z() {
                key.push_str(":Z");
            }
            let c = self.mq(self.cur_moment, g.value());
            self.out.elements.push((key, c));
            if let Some(p) = prev {
                self.line(c, p);
            }
            prev = Some(c);
        }
    }

    fn resolved(&mut self, op: &Instruction, t: &[GateTarget]) {
        let name = op.gate.name;
        match name {
            "MPP" | "SPP" | "SPP_DAG" | "E" | "ELSE_CORRELATED_ERROR" => self.pauli_product(name, t),
            "DETECTOR" | "OBSERVABLE_INCLUDE" | "QUBIT_COORDS" => {}
            "TICK" => self.tick(),
            _ if op.gate.flags & FLAG_TARGETS_PAIRS != 0 => self.two_qubit(name, t),
            _ => {
                self.reserve(t);
                let c = self.mq(self.cur_moment, t[0].value());
                self.out.elements.push((name.to_string(), c));
            }
        }
    }

    fn circuit(&mut self, c: &ir::Circuit) {
        for it in &c.items {
            match it {
                Item::Repeat { body, .. } => {
                    self.nesting += 1;
                    // start_repeat (cur_moment_is_used is never set by this drawer)
                    self.start_next_moment();
                    self.loop_starts.push(self.cur_moment);
                    self.tick_start_moment = self.cur_moment;
                    self.circuit(body);
                    let start = self.loop_starts.pop().unwrap();
                    let (lo, hi) = self.bounds;
                    let pad = 0.5f32 * 3.0 / (2 + self.nesting) as f32;
                    let (x1, x2, y1, y2) = (lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad);
                    let m = self.cur_moment;
                    let mut p = [trans(start, [x1, y1]), trans(start, [x1, y2]), trans(start, [x2, y1]), trans(start, [x2, y2]), trans(m, [x1, y1]), trans(m, [x1, y2]), trans(m, [x2, y1]), trans(m, [x2, y2])];
                    for (k, c) in p.iter_mut().enumerate() {
                        c[0] = (c[0] as f64 + if k < 4 { 0.25 } else { -0.25 }) as f32;
                    }
                    for k in [0, 1, 0, 2, 0, 4, 1, 3, 1, 5, 2, 3, 2, 6, 3, 7, 4, 5, 4, 6, 5, 7, 6, 7] {
                        self.out.red.push(p[k]);
                    }
                    self.start_next_moment();
                    self.tick_start_moment = self.cur_moment;
                    self.nesting -= 1;
                }
                Item::Op(op) => {
                    let f = op.gate.flags;
                    let name = op.gate.name;
                    if matches!(name, "DETECTOR" | "OBSERVABLE_INCLUDE" | "SHIFT_COORDS" | "E" | "ELSE_CORRELATED_ERROR") {
                        if name != "SHIFT_COORDS" {
                            self.resolved(op, &op.targets);
                        }
                    } else if name == "QUBIT_COORDS" {
                    } else if name == "TICK" {
                        self.tick();
                    } else if f & FLAG_TARGETS_COMBINERS != 0 {
                        let paired = f & FLAG_TARGETS_PAIRS != 0;
                        let t = &op.targets;
                        let mut start = 0;
                        while start < t.len() {
                            let mut end = start + 1;
                            while end < t.len() && t[end].is_combiner() {
                                end += 2;
                            }
                            if paired {
                                end += 1;
                                while end < t.len() && t[end].is_combiner() {
                                    end += 2;
                                }
                            }
                            self.resolved(op, &t[start..end.min(t.len())]);
                            start = end;
                        }
                    } else if f & FLAG_TARGETS_PAIRS != 0 {
                        for p in op.targets.chunks(2) {
                            self.resolved(op, p);
                        }
                    } else {
                        for t in &op.targets {
                            self.resolved(op, std::slice::from_ref(t));
                        }
                    }
                }
            }
        }
    }
}

/// Stim's `timeline-3d` diagram of a circuit, as glTF.
pub fn timeline_3d(c: &ir::Circuit) -> Result<String, String> {
    let nq = c.count_qubits() as usize;
    let mut used = BTreeSet::new();
    add_used_qubits(c, &mut used);
    let (coords, bounds) = pick_coords(c, nq, &used);
    let mut d = Timeline3d { out: Basic3d::default(), coords, bounds, cur_moment: 0, tick_start_moment: 0, used: vec![false; nq], has_ticks: c.count_ticks() > 0, loop_starts: Vec::new(), nesting: 0 };
    let (lo, hi) = bounds;
    let y = -2.0 * (lo[0] - 1.0);
    let z = -2.0 * (lo[1] * 0.5 + hi[1] * 0.5);
    for p in [[0.0, y, z], [-3.0, y, z], [-2.5, y - 0.5, z], [-3.0, y, z], [-2.5, y + 0.5, z], [-3.0, y, z]] {
        d.out.red.push(p);
    }
    d.circuit(c);
    for &q in &used {
        let mut p1 = d.mq(0, q);
        p1[0] += 1.0;
        let p2 = d.mq(d.cur_moment + 1, q);
        d.out.lines.push(p1);
        d.out.lines.push(p2);
    }
    d.out.to_gltf()
}

// ---------------------------------------------------------------------------------------------
// The matching graph in 3D (Stim's `dem_match_graph_to_basic_3d_diagram`).

fn flattened_3d(c: &[f64]) -> C3 {
    let mut x = c.first().map_or(0.0, |&v| v as f32);
    let mut y = c.get(1).map_or(0.0, |&v| v as f32);
    let z = c.get(2).map_or(0.0, |&v| v as f32);
    for (k, &v) in c.iter().enumerate().skip(3) {
        let k = k as f64;
        x = (x as f64 + v / k) as f32;
        y = (y as f64 + v / (k * k)) as f32;
        x = (x as f64 + v / (k * k * k)) as f32;
    }
    [x * 3.0, y * 3.0, z * 3.0]
}

/// Stim's `matchgraph-3d` diagram of a detector error model, as glTF.
pub fn matchgraph_3d(dem: &DemProgram) -> Result<String, String> {
    let flat = dem.flattened()?;
    let n = dem.stats().num_detectors as usize;
    let mut det_coords: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for i in &flat.instrs {
        if let DemInstr::Detector { coords, targets, .. } = i {
            for &t in targets {
                det_coords.entry(t as usize).or_insert_with(|| coords.clone());
            }
        }
    }
    let mut coords: Vec<C3> = vec![[0.0; 3]; n];
    let mut min_z = 0f32;
    for (&k, v) in &det_coords {
        if v.is_empty() || k >= n {
            continue;
        }
        coords[k] = flattened_3d(v);
        min_z = if coords[k][2] < min_z { coords[k][2] } else { min_z };
    }
    let (mut nx, mut ny, mut dx, mut dy) = (0f32, 0f32, 1f32, -1f32);
    for (d, c) in coords.iter_mut().enumerate() {
        if det_coords.get(&d).is_none_or(|v| v.is_empty()) {
            *c = [nx * 3.0, ny * 3.0, min_z - 1.0];
            nx += dx;
            ny += dy;
            if ny < 0.0 || nx < 0.0 {
                nx = nx.max(0.0);
                ny = ny.max(0.0);
                dx *= -1.0;
                dy *= -1.0;
            }
        }
    }
    let mut out = Basic3d::default();
    let (lo, hi) = {
        let (mut lo, mut hi) = ([0f32; 3], [0f32; 3]);
        if !coords.is_empty() {
            lo = [f32::INFINITY; 3];
            hi = [f32::NEG_INFINITY; 3];
            for c in &coords {
                for k in 0..3 {
                    lo[k] = if c[k] < lo[k] { c[k] } else { lo[k] };
                    hi[k] = if hi[k] < c[k] { c[k] } else { hi[k] };
                }
            }
        }
        (lo, hi)
    };
    let center = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5];
    let mut boundary_obs: BTreeSet<usize> = BTreeSet::new();
    let mut handle = |targets: &[u64], out: &mut Basic3d| {
        let has_obs = targets.iter().any(|&t| t & OBS != 0);
        let mut pts: Vec<C3> = targets.iter().filter(|&&t| t & OBS == 0).map(|&t| coords[t as usize]).collect();
        if pts.is_empty() {
            return;
        }
        if pts.len() == 1 {
            let mut d = [pts[0][0] - center[0], pts[0][1] - center[1], pts[0][2] - center[2]];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if (r as f64) < 1e-4 {
                d = [1.0, 0.0, 0.0];
            } else {
                d = [d[0] / r, d[1] / r, d[2] / r];
            }
            let a = pts[0];
            pts.push([a[0] + d[0] * 10.0, a[1] + d[1] * 10.0, a[2] + d[2] * 10.0]);
            if has_obs {
                for &t in targets.iter().filter(|&&t| t & OBS == 0) {
                    boundary_obs.insert(t as usize);
                }
            }
        }
        if pts.len() == 2 {
            let list = if has_obs { &mut out.red } else { &mut out.lines };
            list.push(pts[0]);
            list.push(pts[1]);
        } else {
            let mut c = [0f32; 3];
            for e in &pts {
                for k in 0..3 {
                    c[k] += e[k];
                }
            }
            let m = pts.len() as f32;
            c = [c[0] / m, c[1] / m, c[2] / m];
            for e in &pts {
                let list = if has_obs { &mut out.purple } else { &mut out.blue };
                list.push(c);
                list.push(*e);
            }
        }
    };
    for i in &flat.instrs {
        if let DemInstr::Error { pieces, .. } = i {
            for piece in pieces {
                handle(piece, &mut out);
            }
        }
    }
    for (k, c) in coords.iter().enumerate() {
        let excited = boundary_obs.contains(&k);
        out.elements.push((if excited { "EXCITED_DETECTOR" } else { "DETECTOR" }.to_string(), *c));
    }
    out.to_gltf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_as_stim() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"a"), "YQ==");
        assert_eq!(base64(b"ab"), "YWI=");
        assert_eq!(base64(b"abc"), "YWJj");
    }

    #[test]
    fn empty_circuit_draws_the_empty_message() {
        let g = timeline_3d(&ir::Circuit::parse("").unwrap()).unwrap();
        assert!(g.starts_with("{\"accessors\":["));
        assert!(g.contains("\"scene\":0"));
    }
}

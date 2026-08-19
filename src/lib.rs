// The WASM surface is a C ABI: every export takes raw pointers that originate
// from our own `wasm_create_session` and are handed straight back by the JS
// wrapper, which is the only caller. Marking two dozen `extern "C"` entry points
// `unsafe` would say nothing a reader does not already know from the ABI.
#![allow(clippy::not_unsafe_ptr_arg_deref)]
// Several simulate/benchmark calls genuinely take a code, a decoder, a noise
// model and its parameters. Bundling them into a struct would only move the
// argument list somewhere else.
#![allow(clippy::too_many_arguments)]
// Index arithmetic over `round * num_stabs + stabilizer` is the subject matter
// here; iterator adapters obscure it.
#![allow(clippy::needless_range_loop)]

pub mod tableau;
pub mod simulator;
pub mod decoder;
pub mod surface_code;
pub mod circuit_model;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyclass(name = "RotatedSurfaceCode")]
pub struct PyRotatedSurfaceCode {
    code: surface_code::RotatedSurfaceCode,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyRotatedSurfaceCode {
    #[new]
    fn new(d: usize) -> Self {
        PyRotatedSurfaceCode {
            code: surface_code::RotatedSurfaceCode::new(d),
        }
    }

    #[pyo3(signature = (num_rounds, p, bias=None, decoder_type=None))]
    fn simulate(&self, num_rounds: usize, p: f64, bias: Option<f64>, decoder_type: Option<usize>) -> bool {
        self.code.simulate_phenomenological_noise(num_rounds, p, bias.unwrap_or(1.0), decoder_type.unwrap_or(0), 0.0, 0) != 0
    }

    #[pyo3(signature = (p, bias=None, decoder_type=None))]
    fn simulate_data_noise(&self, p: f64, bias: Option<f64>, decoder_type: Option<usize>) -> bool {
        self.code.simulate_data_noise(p, bias.unwrap_or(1.0), decoder_type.unwrap_or(0), 0.0, 0) != 0
    }

    #[getter]
    fn d(&self) -> usize {
        self.code.d
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn stabilizer_qec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRotatedSurfaceCode>()?;
    Ok(())
}

pub struct WasmSession {
    code_type: usize, // 0 for Rotated, 1 for XZZX
    d: usize,
    num_rounds: usize,
    physical_x: Vec<bool>,
    physical_z: Vec<bool>,
    physical_erased: Vec<bool>,
    measurement_errors: Vec<bool>,
    physical_x_u8: Vec<u8>,
    physical_z_u8: Vec<u8>,
    physical_erased_u8: Vec<u8>,
    correction_x: Vec<u8>,
    correction_z: Vec<u8>,
    syndrome: Vec<u8>,
}

/// Seed the generator that every shot draws from.
///
/// Without this the WebAssembly build is deterministic across page loads: the
/// stream starts from the same constant every time, so a reader who reloads
/// sees byte-identical "measurements". Callers should pass something from
/// `crypto.getRandomValues`.
#[cfg(not(feature = "python"))]
#[no_mangle]
pub extern "C" fn wasm_seed(seed_lo: u32, seed_hi: u32) {
    let seed = ((seed_hi as u64) << 32) | (seed_lo as u64);
    surface_code::seed_global_rng(seed);
}

#[no_mangle]
pub extern "C" fn wasm_create_session(d: usize, code_type: usize) -> *mut WasmSession {
    let num_data = d * d;
    let num_stabs = if code_type == 0 {
        let code = surface_code::RotatedSurfaceCode::new(d);
        code.x_stabilizers.len() + code.z_stabilizers.len()
    } else {
        let code = surface_code::XZZXSurfaceCode::new(d);
        code.stabilizers.len()
    };
    let session = Box::new(WasmSession {
        code_type,
        d,
        num_rounds: 1,
        physical_x: vec![false; num_data],
        physical_z: vec![false; num_data],
        physical_erased: vec![false; num_data],
        measurement_errors: vec![false; num_stabs],
        physical_x_u8: vec![0; num_data],
        physical_z_u8: vec![0; num_data],
        physical_erased_u8: vec![0; num_data],
        correction_x: vec![0; num_data],
        correction_z: vec![0; num_data],
        syndrome: vec![0; num_stabs],
    });
    Box::into_raw(session)
}

#[no_mangle]
pub extern "C" fn wasm_set_num_rounds(ptr: *mut WasmSession, num_rounds: usize) {
    let session = unsafe { &mut *ptr };
    session.num_rounds = num_rounds;
    let num_data = session.d * session.d * num_rounds;
    let num_stabs = if session.code_type == 0 {
        let code = surface_code::RotatedSurfaceCode::new(session.d);
        (code.x_stabilizers.len() + code.z_stabilizers.len()) * num_rounds
    } else {
        let code = surface_code::XZZXSurfaceCode::new(session.d);
        code.stabilizers.len() * num_rounds
    };
    session.physical_x.resize(num_data, false);
    session.physical_z.resize(num_data, false);
    session.physical_erased.resize(num_data, false);
    session.measurement_errors.resize(num_stabs, false);
    session.physical_x_u8.resize(num_data, 0);
    session.physical_z_u8.resize(num_data, 0);
    session.physical_erased_u8.resize(num_data, 0);
    session.correction_x.resize(num_data, 0);
    session.correction_z.resize(num_data, 0);
    session.syndrome.resize(num_stabs, 0);
}

#[no_mangle]
pub extern "C" fn wasm_get_physical_x_ptr(ptr: *mut WasmSession) -> *const u8 {
    let session = unsafe { &mut *ptr };
    for i in 0..session.physical_x.len() {
        session.physical_x_u8[i] = if session.physical_x[i] { 1 } else { 0 };
    }
    session.physical_x_u8.as_ptr()
}

#[no_mangle]
pub extern "C" fn wasm_get_physical_z_ptr(ptr: *mut WasmSession) -> *const u8 {
    let session = unsafe { &mut *ptr };
    for i in 0..session.physical_z.len() {
        session.physical_z_u8[i] = if session.physical_z[i] { 1 } else { 0 };
    }
    session.physical_z_u8.as_ptr()
}

#[no_mangle]
pub extern "C" fn wasm_get_physical_erased_ptr(ptr: *mut WasmSession) -> *const u8 {
    let session = unsafe { &mut *ptr };
    for i in 0..session.physical_erased.len() {
        session.physical_erased_u8[i] = if session.physical_erased[i] { 1 } else { 0 };
    }
    session.physical_erased_u8.as_ptr()
}

#[no_mangle]
pub extern "C" fn wasm_get_correction_x_ptr(ptr: *mut WasmSession) -> *const u8 {
    let session = unsafe { &*ptr };
    session.correction_x.as_ptr()
}

#[no_mangle]
pub extern "C" fn wasm_get_correction_z_ptr(ptr: *mut WasmSession) -> *const u8 {
    let session = unsafe { &*ptr };
    session.correction_z.as_ptr()
}

#[no_mangle]
pub extern "C" fn wasm_free_session(ptr: *mut WasmSession) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

#[no_mangle]
pub extern "C" fn wasm_toggle_error(ptr: *mut WasmSession, q_idx: usize, error_type: usize, t: usize) {
    let session = unsafe { &mut *ptr };
    let num_data = session.d * session.d;
    let idx = q_idx + t * num_data;
    if idx < session.physical_x.len() {
        if error_type == 0 {
            session.physical_x[idx] ^= true;
        } else {
            session.physical_z[idx] ^= true;
        }
    }
}

#[no_mangle]
pub extern "C" fn wasm_toggle_erasure(ptr: *mut WasmSession, q_idx: usize, t: usize) {
    let session = unsafe { &mut *ptr };
    let num_data = session.d * session.d;
    let idx = q_idx + t * num_data;
    if idx < session.physical_erased.len() {
        session.physical_erased[idx] ^= true;
    }
}

#[no_mangle]
pub extern "C" fn wasm_toggle_measurement_error(ptr: *mut WasmSession, s_idx: usize, t: usize) {
    let session = unsafe { &mut *ptr };
    let num_stabs = if session.code_type == 0 {
        let code = surface_code::RotatedSurfaceCode::new(session.d);
        code.x_stabilizers.len() + code.z_stabilizers.len()
    } else {
        let code = surface_code::XZZXSurfaceCode::new(session.d);
        code.stabilizers.len()
    };
    let idx = s_idx + t * num_stabs;
    if idx < session.measurement_errors.len() {
        session.measurement_errors[idx] ^= true;
    }
}

#[no_mangle]
pub extern "C" fn wasm_clear_errors(ptr: *mut WasmSession) {
    let session = unsafe { &mut *ptr };
    for val in &mut session.physical_x { *val = false; }
    for val in &mut session.physical_z { *val = false; }
    for val in &mut session.physical_erased { *val = false; }
    for val in &mut session.measurement_errors { *val = false; }
}

#[no_mangle]
pub extern "C" fn wasm_get_data_qubit_count(ptr: *mut WasmSession) -> usize {
    let session = unsafe { &*ptr };
    session.d * session.d
}

#[no_mangle]
pub extern "C" fn wasm_get_stabilizer_count(ptr: *mut WasmSession) -> usize {
    let session = unsafe { &*ptr };
    if session.code_type == 0 {
        let code = surface_code::RotatedSurfaceCode::new(session.d);
        code.x_stabilizers.len() + code.z_stabilizers.len()
    } else {
        let code = surface_code::XZZXSurfaceCode::new(session.d);
        code.stabilizers.len()
    }
}

#[no_mangle]
pub extern "C" fn wasm_get_stabilizer_type(ptr: *mut WasmSession, idx: usize) -> usize {
    let session = unsafe { &*ptr };
    if session.code_type == 0 {
        let code = surface_code::RotatedSurfaceCode::new(session.d);
        let num_x = code.x_stabilizers.len();
        if idx < num_x {
            1 // X stabilizer
        } else {
            0 // Z stabilizer
        }
    } else {
        2 // XZZX stabilizer
    }
}

#[no_mangle]
pub extern "C" fn wasm_get_syndrome(ptr: *mut WasmSession) -> *const u8 {
    let session = unsafe { &mut *ptr };
    let d = session.d;
    let num_rounds = session.num_rounds;
    let num_data = d * d;
    
    if session.code_type == 0 {
        let code = surface_code::RotatedSurfaceCode::new(d);
        let num_x = code.x_stabilizers.len();
        let num_z = code.z_stabilizers.len();
        let num_stabs = num_x + num_z;
        
        for t in 0..num_rounds {
            for s_idx in 0..num_z {
                let neighbors = code.get_neighbors(&code.z_stabilizers[s_idx]);
                let mut parity = 0;
                // Accumulate physical X errors up to round t
                for t_prime in 0..=t {
                    for &q in &neighbors {
                        if session.physical_x[q + t_prime * num_data] {
                            parity ^= 1;
                        }
                    }
                }
                if session.measurement_errors[num_x + s_idx + t * num_stabs] {
                    parity ^= 1;
                }
                session.syndrome[num_x + s_idx + t * num_stabs] = parity;
            }

            for s_idx in 0..num_x {
                let neighbors = code.get_neighbors(&code.x_stabilizers[s_idx]);
                let mut parity = 0;
                // Accumulate physical Z errors up to round t
                for t_prime in 0..=t {
                    for &q in &neighbors {
                        if session.physical_z[q + t_prime * num_data] {
                            parity ^= 1;
                        }
                    }
                }
                if session.measurement_errors[s_idx + t * num_stabs] {
                    parity ^= 1;
                }
                session.syndrome[s_idx + t * num_stabs] = parity;
            }
        }
    } else {
        let code = surface_code::XZZXSurfaceCode::new(d);
        let num_stabs = code.stabilizers.len();
        
        for t in 0..num_rounds {
            for s_idx in 0..num_stabs {
                let (sx, sy) = code.stabilizers[s_idx];
                let mut parity = 0;
                for t_prime in 0..=t {
                    if let Some(q) = code.get_neighbor_idx(sx as i32 - 1, sy as i32 - 1) {
                        if session.physical_z[q + t_prime * num_data] { parity ^= 1; }
                    }
                    if let Some(q) = code.get_neighbor_idx(sx as i32 + 1, sy as i32 - 1) {
                        if session.physical_x[q + t_prime * num_data] { parity ^= 1; }
                    }
                    if let Some(q) = code.get_neighbor_idx(sx as i32 - 1, sy as i32 + 1) {
                        if session.physical_x[q + t_prime * num_data] { parity ^= 1; }
                    }
                    if let Some(q) = code.get_neighbor_idx(sx as i32 + 1, sy as i32 + 1) {
                        if session.physical_z[q + t_prime * num_data] { parity ^= 1; }
                    }
                }
                if session.measurement_errors[s_idx + t * num_stabs] {
                    parity ^= 1;
                }
                session.syndrome[s_idx + t * num_stabs] = parity;
            }
        }
    }
    session.syndrome.as_ptr()
}

#[no_mangle]
pub extern "C" fn wasm_decode(
    ptr: *mut WasmSession,
    decoder_type: usize,
) -> u8 {
    let session = unsafe { &mut *ptr };
    let d = session.d;
    let num_rounds = session.num_rounds;
    let num_data = d * d;

    for val in &mut session.correction_x {
        *val = 0;
    }
    for val in &mut session.correction_z {
        *val = 0;
    }

    // Call wasm_get_syndrome to populate session.syndrome first
    wasm_get_syndrome(ptr);

    if session.code_type == 0 {
        let code = surface_code::RotatedSurfaceCode::new(d);
        let num_x = code.x_stabilizers.len();
        let num_z = code.z_stabilizers.len();
        let num_stabs = num_x + num_z;

        // --- Z stabilizers (X errors) ---
        let graph_z = code.build_syndrome_graph(num_rounds, true);
        let mut defects_z = vec![false; graph_z.num_nodes];
        for t in 0..num_rounds {
            for s_idx in 0..num_z {
                let outcome = session.syndrome[num_x + s_idx + t * num_stabs] == 1;
                let prev_outcome = if t == 0 { false } else { session.syndrome[num_x + s_idx + (t - 1) * num_stabs] == 1 };
                defects_z[s_idx + t * num_z] = outcome ^ prev_outcome;
            }
        }
        let mut erased_edges_z = vec![false; graph_z.edges.len()];
        for edge_idx in 0..graph_z.edges.len() {
            if let Some(q_idx) = graph_z.edge_to_qubit[edge_idx] {
                let u = graph_z.edges[edge_idx].u;
                let t_layer = u / num_z;
                let idx = q_idx + t_layer * num_data;
                if idx < session.physical_erased.len() && session.physical_erased[idx] {
                    erased_edges_z[edge_idx] = true;
                }
            }
        }
        let correction_z_edges = match decoder_type {
            1 => decoder::decode_greedy(&graph_z, &defects_z, &erased_edges_z),
            2 => decoder::decode_mwpm(&graph_z, &defects_z, &erased_edges_z),
            _ => decoder::decode_union_find(&graph_z, &defects_z, &erased_edges_z),
        };
        for edge_idx in correction_z_edges {
            if let Some(q_idx) = graph_z.edge_to_qubit[edge_idx] {
                let u = graph_z.edges[edge_idx].u;
                let t_layer = u / num_z;
                if t_layer < num_rounds {
                    session.correction_x[q_idx + t_layer * num_data] ^= 1;
                }
            }
        }

        // --- X stabilizers (Z errors) ---
        let graph_x = code.build_syndrome_graph(num_rounds, false);
        let mut defects_x = vec![false; graph_x.num_nodes];
        for t in 0..num_rounds {
            for s_idx in 0..num_x {
                let outcome = session.syndrome[s_idx + t * num_stabs] == 1;
                let prev_outcome = if t == 0 { false } else { session.syndrome[s_idx + (t - 1) * num_stabs] == 1 };
                defects_x[s_idx + t * num_x] = outcome ^ prev_outcome;
            }
        }
        let mut erased_edges_x = vec![false; graph_x.edges.len()];
        for edge_idx in 0..graph_x.edges.len() {
            if let Some(q_idx) = graph_x.edge_to_qubit[edge_idx] {
                let u = graph_x.edges[edge_idx].u;
                let t_layer = u / num_x;
                let idx = q_idx + t_layer * num_data;
                if idx < session.physical_erased.len() && session.physical_erased[idx] {
                    erased_edges_x[edge_idx] = true;
                }
            }
        }
        let correction_x_edges = match decoder_type {
            1 => decoder::decode_greedy(&graph_x, &defects_x, &erased_edges_x),
            2 => decoder::decode_mwpm(&graph_x, &defects_x, &erased_edges_x),
            _ => decoder::decode_union_find(&graph_x, &defects_x, &erased_edges_x),
        };
        for edge_idx in correction_x_edges {
            if let Some(q_idx) = graph_x.edge_to_qubit[edge_idx] {
                let u = graph_x.edges[edge_idx].u;
                let t_layer = u / num_x;
                if t_layer < num_rounds {
                    session.correction_z[q_idx + t_layer * num_data] ^= 1;
                }
            }
        }

        // We check logical errors at the final round by accumulating physical and correction operators
        let mut accumulated_x = vec![false; num_data];
        let mut accumulated_z = vec![false; num_data];
        for t in 0..num_rounds {
            for q in 0..num_data {
                accumulated_x[q] ^= session.physical_x[q + t * num_data] ^ (session.correction_x[q + t * num_data] != 0);
                accumulated_z[q] ^= session.physical_z[q + t * num_data] ^ (session.correction_z[q + t * num_data] != 0);
            }
        }

        let mut logical_x = false;
        for y_idx in 0..d {
            let q_idx = d * y_idx; // the leftmost column: 0, d, 2d, ...
            if accumulated_x[q_idx] {
                logical_x ^= true;
            }
        }

        let mut logical_z = false;
        for x_idx in 0..d {
            let q_idx = x_idx; // the top row: 0, 1, 2, ..., d-1
            if accumulated_z[q_idx] {
                logical_z ^= true;
            }
        }

        if logical_x || logical_z { 1 } else { 0 }
    } else {
        let code = surface_code::XZZXSurfaceCode::new(d);
        let num_stabs = code.stabilizers.len();

        // One matching over both error families. Decoding the same defect set
        // independently on the X graph and the Z graph — which is what
        // `defects_x = defects_z.clone()` amounted to — explains every defect
        // twice and applies both corrections, so a single X error came back
        // with the correct X correction plus invented Z ones.
        let (graph, edge_is_x) = code.build_combined_graph(num_rounds);

        let mut defects = vec![false; graph.num_nodes];
        for t in 0..num_rounds {
            for s_idx in 0..num_stabs {
                let outcome = session.syndrome[s_idx + t * num_stabs] == 1;
                let prev_outcome = if t == 0 { false } else { session.syndrome[s_idx + (t - 1) * num_stabs] == 1 };
                defects[s_idx + t * num_stabs] = outcome ^ prev_outcome;
            }
        }

        let mut erased_edges = vec![false; graph.edges.len()];
        for edge_idx in 0..graph.edges.len() {
            if let Some(q_idx) = graph.edge_to_qubit[edge_idx] {
                let u = graph.edges[edge_idx].u;
                let t_layer = u / num_stabs;
                let idx = q_idx + t_layer * num_data;
                if idx < session.physical_erased.len() && session.physical_erased[idx] {
                    erased_edges[edge_idx] = true;
                }
            }
        }

        let correction_edges = match decoder_type {
            1 => decoder::decode_greedy(&graph, &defects, &erased_edges),
            2 => decoder::decode_mwpm(&graph, &defects, &erased_edges),
            _ => decoder::decode_union_find(&graph, &defects, &erased_edges),
        };
        for edge_idx in correction_edges {
            if let Some(q_idx) = graph.edge_to_qubit[edge_idx] {
                let u = graph.edges[edge_idx].u;
                let t_layer = u / num_stabs;
                if t_layer < num_rounds {
                    if edge_is_x[edge_idx] {
                        session.correction_x[q_idx + t_layer * num_data] ^= 1;
                    } else {
                        session.correction_z[q_idx + t_layer * num_data] ^= 1;
                    }
                }
            }
        }

        let mut accumulated_x = vec![false; num_data];
        let mut accumulated_z = vec![false; num_data];
        for t in 0..num_rounds {
            for q in 0..num_data {
                accumulated_x[q] ^= session.physical_x[q + t * num_data] ^ (session.correction_x[q + t * num_data] != 0);
                accumulated_z[q] ^= session.physical_z[q + t * num_data] ^ (session.correction_z[q + t * num_data] != 0);
            }
        }

        // Ask the stabilizer group directly rather than trusting a hard-coded
        // logical string. See surface_code::LogicalCheck.
        let (mut rx, mut rz) = (0u128, 0u128);
        for q in 0..num_data {
            if accumulated_x[q] { rx |= 1u128 << q; }
            if accumulated_z[q] { rz |= 1u128 << q; }
        }
        if code.logical.is_logical(rx, rz) { 1 } else { 0 }
    }
}

static mut FIDELITY_RESULTS: [f64; 3] = [0.0; 3];

/// Diagonal of the logical Pauli-transfer matrix.
///
/// Under Pauli noise and a Pauli decoder the effective logical channel is
/// itself a Pauli channel,
///
/// ```text
/// rho -> p_I rho + p_X X rho X + p_Y Y rho Y + p_Z Z rho Z
/// ```
///
/// which acts on a Bloch vector by shrinking each axis independently:
///
/// ```text
/// rx' = rx (p_I + p_X - p_Y - p_Z)
/// ry' = ry (p_I - p_X + p_Y - p_Z)
/// rz' = rz (p_I - p_X - p_Y + p_Z)
/// ```
///
/// Those three factors are what this returns, in that order. They are a real
/// characterisation of the channel: each lies in [-1, 1], and a noiseless
/// channel gives (1, 1, 1) because it shrinks nothing.
///
/// The previous version ran three *identical* simulations — for data and
/// phenomenological noise the three calls differed in nothing but the variable
/// they were assigned to — and reported `1 - 2 * failure_rate` for each as
/// though they were Bloch components. At zero noise that produced the vector
/// (1, 1, 1) interpreted as a *state*, which has length sqrt(3) and lies
/// outside the Bloch sphere.
///
/// Circuit-level noise still reports only whether the one prepared logical
/// operator flipped, so its X and Z classes cannot be separated; for that mode
/// all three factors collapse to the same number.
#[no_mangle]
pub extern "C" fn wasm_estimate_logical_fidelity(
    d: usize,
    code_type: usize,
    decoder_type: usize,
    p: f64,
    bias: f64,
    noise_mode: usize,
    num_rounds: usize,
    runs: usize,
    erasure_rate: f64,
    correlated_noise: usize,
) -> *const f64 {
    // Counts of the logical Pauli class left behind: index 0 = I, 1 = X,
    // 2 = Z, 3 = Y, matching the bitmask the simulators return.
    let mut classes = [0usize; 4];

    if code_type == 0 {
        let code = surface_code::RotatedSurfaceCode::new(d);
        // The detector error model depends only on the code and the round
        // count, so it is built once here rather than per shot.
        let model = (noise_mode == 2)
            .then(|| circuit_model::build(&code.circuit_layout(), num_rounds));
        for _ in 0..runs {
            let outcome = match noise_mode {
                0 => code.simulate_data_noise(p, bias, decoder_type, erasure_rate, correlated_noise),
                1 => code.simulate_phenomenological_noise(num_rounds, p, bias, decoder_type, erasure_rate, correlated_noise),
                2 => code.simulate_circuit_noise_with_model(
                    model.as_ref().unwrap(), num_rounds, p, bias, "zero",
                    decoder_type, erasure_rate, correlated_noise),
                _ => 0,
            };
            classes[(outcome & 3) as usize] += 1;
        }
    } else {
        let code = surface_code::XZZXSurfaceCode::new(d);
        // One graph, not two: an XZZX plaquette reads X Z Z X from a single
        // ancilla, so both error families fire the same detectors.
        let model = (noise_mode == 2)
            .then(|| circuit_model::build_combined(&code.circuit_layout(), num_rounds));
        for _ in 0..runs {
            let outcome = match noise_mode {
                0 => code.simulate_data_noise(p, bias, decoder_type, erasure_rate, correlated_noise),
                1 => code.simulate_phenomenological_noise(num_rounds, p, bias, decoder_type, erasure_rate, correlated_noise),
                2 => code.simulate_circuit_noise_with_model(
                    model.as_ref().unwrap(), num_rounds, p, bias,
                    decoder_type, erasure_rate, correlated_noise),
                _ => 0,
            };
            classes[(outcome & 3) as usize] += 1;
        }
    }

    let total = runs.max(1) as f64;
    let p_i = classes[0] as f64 / total;
    let p_x = classes[1] as f64 / total;
    let p_z = classes[2] as f64 / total;
    let p_y = classes[3] as f64 / total;

    unsafe {
        FIDELITY_RESULTS[0] = p_i + p_x - p_y - p_z;
        FIDELITY_RESULTS[1] = p_i - p_x + p_y - p_z;
        FIDELITY_RESULTS[2] = p_i - p_x - p_y + p_z;
        std::ptr::addr_of!(FIDELITY_RESULTS) as *const f64
    }
}

#[cfg(not(feature = "python"))]
static mut FAULT_TEST: [f64; 2] = [0.0; 2];

/// Exhaustive single-fault check for the circuit model. Returns [tested, failed].
#[cfg(not(feature = "python"))]
#[no_mangle]
pub extern "C" fn wasm_circuit_single_fault_test(
    d: usize,
    num_rounds: usize,
    decoder_type: usize,
) -> *const f64 {
    let code = surface_code::RotatedSurfaceCode::new(d);
    let layout = code.circuit_layout();
    let model = circuit_model::build(&layout, num_rounds);

    // Logical Z runs down a column, logical X across a row; a residual X is a
    // logical X when its parity against the column is odd, and vice versa.
    let mut col = 0u128;
    let mut row = 0u128;
    for i in 0..d {
        col |= 1u128 << (i * d);
        row |= 1u128 << i;
    }

    let (tested, failed) =
        circuit_model::single_fault_failures(&layout, &model, num_rounds, decoder_type, row, col);
    unsafe {
        FAULT_TEST[0] = tested as f64;
        FAULT_TEST[1] = failed as f64;
        std::ptr::addr_of!(FAULT_TEST) as *const f64
    }
}

#[cfg(not(feature = "python"))]
static mut MODEL_STATS: [f64; 12] = [0.0; 12];

/// Diagnostics for the circuit detector error model.
#[cfg(not(feature = "python"))]
#[no_mangle]
pub extern "C" fn wasm_circuit_model_stats(d: usize, num_rounds: usize) -> *const f64 {
    let code = surface_code::RotatedSurfaceCode::new(d);
    let st = circuit_model::stats(&code.circuit_layout(), num_rounds);
    unsafe {
        for i in 0..5 {
            MODEL_STATS[i] = st.x_buckets[i] as f64;
            MODEL_STATS[5 + i] = st.z_buckets[i] as f64;
        }
        MODEL_STATS[10] = st.x_edges as f64;
        MODEL_STATS[11] = st.z_edges as f64;
        std::ptr::addr_of!(MODEL_STATS) as *const f64
    }
}

#[no_mangle]
pub extern "C" fn wasm_run_benchmark(
    d: usize,
    code_type: usize,
    decoder_type: usize,
    p: f64,
    bias: f64,
    num_rounds: usize,
    num_runs: usize,
    noise_mode: usize,
    erasure_rate: f64,
    correlated_noise: usize,
) -> f64 {
    let mut failures = 0;

    if code_type == 0 {
        let code = surface_code::RotatedSurfaceCode::new(d);
        // Built once: the model is a property of the circuit, not of a shot.
        let model = (noise_mode == 2)
            .then(|| circuit_model::build(&code.circuit_layout(), num_rounds));
        for _ in 0..num_runs {
            let failed = match noise_mode {
                0 => code.simulate_data_noise(p, bias, decoder_type, erasure_rate, correlated_noise),
                1 => code.simulate_phenomenological_noise(num_rounds, p, bias, decoder_type, erasure_rate, correlated_noise),
                2 => code.simulate_circuit_noise_with_model(
                    model.as_ref().unwrap(), num_rounds, p, bias, "zero",
                    decoder_type, erasure_rate, correlated_noise),
                _ => 0,
            };
            if failed != 0 {
                failures += 1;
            }
        }
    } else {
        let code = surface_code::XZZXSurfaceCode::new(d);
        // One graph, not two: an XZZX plaquette reads X Z Z X from a single
        // ancilla, so both error families fire the same detectors.
        let model = (noise_mode == 2)
            .then(|| circuit_model::build_combined(&code.circuit_layout(), num_rounds));
        for _ in 0..num_runs {
            let failed = match noise_mode {
                0 => code.simulate_data_noise(p, bias, decoder_type, erasure_rate, correlated_noise),
                1 => code.simulate_phenomenological_noise(num_rounds, p, bias, decoder_type, erasure_rate, correlated_noise),
                2 => code.simulate_circuit_noise_with_model(
                    model.as_ref().unwrap(), num_rounds, p, bias,
                    decoder_type, erasure_rate, correlated_noise),
                _ => 0,
            };
            if failed != 0 {
                failures += 1;
            }
        }
    }

    (failures as f64) / (num_runs as f64)
}

#[no_mangle]
pub extern "C" fn wasm_get_data_qubit_x(ptr: *mut WasmSession, idx: usize) -> usize {
    let session = unsafe { &*ptr };
    if session.code_type == 0 {
        let code = surface_code::RotatedSurfaceCode::new(session.d);
        if idx < code.data_qubits.len() {
            code.data_qubits[idx].0
        } else {
            0
        }
    } else {
        let code = surface_code::XZZXSurfaceCode::new(session.d);
        if idx < code.data_qubits.len() {
            code.data_qubits[idx].0
        } else {
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn wasm_get_data_qubit_y(ptr: *mut WasmSession, idx: usize) -> usize {
    let session = unsafe { &*ptr };
    if session.code_type == 0 {
        let code = surface_code::RotatedSurfaceCode::new(session.d);
        if idx < code.data_qubits.len() {
            code.data_qubits[idx].1
        } else {
            0
        }
    } else {
        let code = surface_code::XZZXSurfaceCode::new(session.d);
        if idx < code.data_qubits.len() {
            code.data_qubits[idx].1
        } else {
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn wasm_get_stabilizer_x(ptr: *mut WasmSession, idx: usize) -> usize {
    let session = unsafe { &*ptr };
    if session.code_type == 0 {
        let code = surface_code::RotatedSurfaceCode::new(session.d);
        let num_x = code.x_stabilizers.len();
        if idx < num_x {
            code.x_stabilizers[idx].0
        } else if idx < num_x + code.z_stabilizers.len() {
            code.z_stabilizers[idx - num_x].0
        } else {
            0
        }
    } else {
        let code = surface_code::XZZXSurfaceCode::new(session.d);
        if idx < code.stabilizers.len() {
            code.stabilizers[idx].0
        } else {
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn wasm_get_stabilizer_y(ptr: *mut WasmSession, idx: usize) -> usize {
    let session = unsafe { &*ptr };
    if session.code_type == 0 {
        let code = surface_code::RotatedSurfaceCode::new(session.d);
        let num_x = code.x_stabilizers.len();
        if idx < num_x {
            code.x_stabilizers[idx].1
        } else if idx < num_x + code.z_stabilizers.len() {
            code.z_stabilizers[idx - num_x].1
        } else {
            0
        }
    } else {
        let code = surface_code::XZZXSurfaceCode::new(session.d);
        if idx < code.stabilizers.len() {
            code.stabilizers[idx].1
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::simulator::StabilizerSimulator;

    /// The extraction circuit must be a valid *simultaneous* measurement of all
    /// plaquettes, not merely a correct one plaquette at a time.
    ///
    /// This cannot be checked on the Pauli frame. Frame simulation presumes the
    /// circuit projects onto a stabilizer eigenspace and only tracks flips
    /// relative to that; if the schedule made neighbouring plaquettes disturb
    /// each other the frame picture would stay self-consistent while the real
    /// device produced noise. So it is checked against the tableau: once the
    /// first round has projected, every later noiseless round must reproduce its
    /// outcomes exactly.
    #[test]
    fn xzzx_extraction_circuit_measures_commuting_stabilizers() {
        use crate::circuit_model::Op;
        for d in [3usize, 5, 7] {
            let code = crate::surface_code::XZZXSurfaceCode::new(d);
            let program = code.round_program();
            let num_stabs = code.stabilizers.len();
            let n = code.data_qubits.len() + num_stabs;

            for seed in [1u64, 2, 3, 4, 5] {
                let mut sim = StabilizerSimulator::with_seed(n, seed);
                let mut rounds: Vec<Vec<u8>> = Vec::new();
                for _ in 0..4 {
                    let mut out = vec![0u8; num_stabs];
                    for &op in &program {
                        match op {
                            Op::Reset(q) => { if sim.measure_z(q) == 1 { sim.apply_x(q); } }
                            Op::H(q) => sim.apply_h(q),
                            Op::Cnot(c, t) => sim.apply_cnot(c, t),
                            Op::Cz(a, b) => { sim.apply_h(b); sim.apply_cnot(a, b); sim.apply_h(b); }
                            Op::Measure(q, _, idx) => out[idx] = sim.measure_z(q),
                            Op::Noise(_) => {}
                        }
                    }
                    rounds.push(out);
                }
                assert_eq!(rounds[1], rounds[2], "d={d} seed={seed}: round 2 != round 3");
                assert_eq!(rounds[2], rounds[3], "d={d} seed={seed}: round 3 != round 4");
            }
        }
    }

    #[test]
    #[ignore]
    fn count_parallel_edges() {
        use crate::circuit_model::*;
        use std::collections::HashMap;
        for d in [3usize, 5] {
            // XZZX combined graph
            let code = crate::surface_code::XZZXSurfaceCode::new(d);
            let layout = code.circuit_layout();
            let m = build_combined(&layout, d);
            let mut by_pair: HashMap<(usize, usize), usize> = HashMap::new();
            for e in &m.graph.graph.edges {
                *by_pair.entry((e.u.min(e.v), e.u.max(e.v))).or_insert(0) += 1;
            }
            let par = by_pair.values().filter(|&&c| c > 1).count();
            let worst = by_pair.values().max().unwrap();
            println!("XZZX    d={d}: {} edges, {} node-pairs, {} pairs with >1 edge, worst {}",
                m.graph.graph.edges.len(), by_pair.len(), par, worst);

            // rotated, both graphs
            let rc = crate::surface_code::RotatedSurfaceCode::new(d);
            let rl = rc.circuit_layout();
            let rm = build(&rl, d);
            for (name, g) in [("for_x", &rm.for_x_errors), ("for_z", &rm.for_z_errors)] {
                let mut bp: HashMap<(usize, usize), usize> = HashMap::new();
                for e in &g.graph.edges { *bp.entry((e.u.min(e.v), e.u.max(e.v))).or_insert(0) += 1; }
                let p2 = bp.values().filter(|&&c| c > 1).count();
                println!("rotated d={d} {name}: {} edges, {} node-pairs, {} pairs with >1 edge, worst {}",
                    g.graph.edges.len(), bp.len(), p2, bp.values().max().unwrap());
            }
        }
    }

    /// `classify` must agree with the rotated code's own hand-derived logicals.
    ///
    /// That code knows its answer independently — logical X is a column of X,
    /// logical Z a row of Z — so it is the one place the derived
    /// representatives can be checked against something not derived the same
    /// way. If the null-space search picked the wrong pair, or labelled X as Z,
    /// this disagrees.
    #[test]
    fn logical_classification_matches_the_rotated_code() {
        for d in [3usize, 5] {
            let code = crate::surface_code::RotatedSurfaceCode::new(d);
            let n = d * d;
            let mut ops: Vec<(u128, u128)> = Vec::new();
            for st in &code.x_stabilizers {
                let mut px = 0u128;
                for &(dx, dy) in &[(-1i32, -1i32), (-1, 1), (1, -1), (1, 1)] {
                    if let Some(q) = code.neighbor_at(st, dx, dy) { px |= 1u128 << q; }
                }
                ops.push((px, 0));
            }
            for st in &code.z_stabilizers {
                let mut pz = 0u128;
                for &(dx, dy) in &[(-1i32, -1i32), (-1, 1), (1, -1), (1, 1)] {
                    if let Some(q) = code.neighbor_at(st, dx, dy) { pz |= 1u128 << q; }
                }
                ops.push((0, pz));
            }
            let check = crate::surface_code::LogicalCheck::new(&ops, n);

            // The code's own convention, from its simulators.
            let mut column = 0u128; // logical X support
            let mut row = 0u128;    // logical Z support
            for i in 0..d {
                column |= 1u128 << (i * d);
                row |= 1u128 << i;
            }

            let mut rng: u64 = 0x2545F4914F6CDD1D;
            let mut checked = 0usize;
            for _ in 0..4000 {
                let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
                let mask = if n >= 128 { u128::MAX } else { (1u128 << n) - 1 };
                let rx = ((next() as u128) | ((next() as u128) << 64)) & mask;
                let rz = ((next() as u128) | ((next() as u128) << 64)) & mask;

                let want = ((rx & column).count_ones() % 2) as u8
                    | (((rz & row).count_ones() % 2) as u8) << 1;
                let got = check.classify(rx, rz);
                assert_eq!(got, want, "d={d}: classify disagreed on rx={rx:b} rz={rz:b}");
                checked += 1;
            }
            assert!(checked > 1000);
        }
    }

    /// The frame shadowing the rotated circuit must stay in step with it.
    ///
    /// The class is now read off a Pauli frame carried alongside the tableau. If
    /// any gate, injected Pauli or applied correction failed to reach the frame,
    /// the two would drift apart silently — the shot would still "work" and just
    /// report the wrong logical class. Two things pin it down: a noiseless
    /// circuit must leave the frame clean, and under noise the two logical types
    /// must occur at comparable rates, since at bias 0.5 nothing distinguishes
    /// X from Z.
    #[test]
    fn rotated_circuit_frame_tracks_the_tableau() {
        let d = 5usize;
        let code = crate::surface_code::RotatedSurfaceCode::new(d);
        let layout = code.circuit_layout();
        let model = crate::circuit_model::build(&layout, d);

        for _ in 0..200 {
            let clean = code.simulate_circuit_noise_with_model(
                &model, d, 0.0, 0.5, "zero", 0, 0.0, 0);
            assert_eq!(clean, 0, "a noiseless circuit left something in the frame");
        }

        let (mut only_x, mut only_z, mut both, mut n) = (0usize, 0usize, 0usize, 0usize);
        for _ in 0..4000 {
            let c = code.simulate_circuit_noise_with_model(
                &model, d, 0.006, 0.5, "zero", 0, 0.0, 0);
            match c { 0 => {}, 1 => only_x += 1, 2 => only_z += 1, _ => both += 1 }
            if c != 0 { n += 1 }
        }
        assert!(n > 40, "too few failures to judge: {n}");
        let x = only_x + both;
        let z = only_z + both;
        let ratio = x as f64 / z.max(1) as f64;
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "logical X and Z should be comparable at bias 0.5, got X={x} Z={z} (ratio {ratio:.2})"
        );
    }

    /// The derived logical representatives must actually be logical operators.
    ///
    /// `find_logical_pair` packs a Pauli into 2n bits of a u128, so a code with
    /// n > 64 data qubits — XZZX from d = 9 up — would silently run off the end
    /// of the word. Wrong representatives do not announce themselves: the shot
    /// still returns a class, just the wrong one, and the rate can look entirely
    /// plausible. So check the defining properties directly.
    #[test]
    fn derived_logicals_are_valid_at_every_distance() {
        for d in [3usize, 5, 7, 9, 11] {
            let code = crate::surface_code::XZZXSurfaceCode::new(d);
            let n = d * d;
            let (lx, lz) = code.logical.representatives();

            // Rebuild the stabilizer set the way the constructor does.
            let index_of = |x: i32, y: i32| -> Option<usize> {
                if x >= 1 && x < (2 * d) as i32 && y >= 1 && y < (2 * d) as i32 {
                    Some(((x - 1) as usize / 2) + d * ((y - 1) as usize / 2))
                } else { None }
            };
            let mut ops: Vec<(u128, u128)> = Vec::new();
            for &(sx, sy) in &code.stabilizers {
                let (mut px, mut pz) = (0u128, 0u128);
                for &(dx, dy, is_x) in &[(-1i32,-1i32,true), (1,1,true), (1,-1,false), (-1,1,false)] {
                    if let Some(q) = index_of(sx as i32 + dx, sy as i32 + dy) {
                        if is_x { px |= 1u128 << q; } else { pz |= 1u128 << q; }
                    }
                }
                ops.push((px, pz));
            }

            let commutes = |a: (u128, u128), b: (u128, u128)| {
                ((a.0 & b.1).count_ones() + (a.1 & b.0).count_ones()) % 2 == 0
            };
            assert!(lx != (0, 0) && lz != (0, 0), "d={d}: no logical pair found (n={n})");
            for (i, &st) in ops.iter().enumerate() {
                assert!(commutes(lx, st), "d={d}: logical_x anticommutes with stabilizer {i}");
                assert!(commutes(lz, st), "d={d}: logical_z anticommutes with stabilizer {i}");
            }
            assert!(!commutes(lx, lz), "d={d}: the two representatives commute");
            assert!(code.logical.is_logical(lx.0, lx.1), "d={d}: logical_x is a stabilizer product");
            assert!(code.logical.is_logical(lz.0, lz.1), "d={d}: logical_z is a stabilizer product");
        }
    }

    #[test]
    fn union_find_is_deterministic() {
        use crate::decoder::decode_union_find;
        let d = 9usize;
        let code = crate::surface_code::RotatedSurfaceCode::new(d);
        let graph = code.build_syndrome_graph(d, true);
        let none = vec![false; graph.edges.len()];
        let mut rng: u64 = 0xDEADBEEF12345678;
        let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
        for trial in 0..50 {
            let mut defects = vec![false; graph.num_nodes];
            for v in defects.iter_mut() {
                if ((next() >> 11) as f64 / 9007199254740992.0) < 0.1 { *v = true; }
            }
            let a = decode_union_find(&graph, &defects, &none);
            let b = decode_union_find(&graph, &defects, &none);
            assert_eq!(a.len(), b.len(), "trial {trial}: union-find gave {} then {} edges", a.len(), b.len());
        }
    }

    /// The matching decoder must never return a heavier correction than the
    /// cheaper decoders it is supposed to beat.
    ///
    /// It used to. Above sixteen defects it handed the problem to the greedy
    /// decoder without saying so, and above 50,000 search steps it did the same
    /// — so at d = 9 phenomenological, where both limits are exceeded on nearly
    /// every shot, asking for the best decoder silently gave you the worst. It
    /// showed up as "exact MWPM" scoring below Union-Find and reporting a
    /// threshold four times too low.
    #[test]
    fn matching_decoder_is_never_heavier_than_the_others() {
        use crate::decoder::{decode_greedy, decode_mwpm, decode_union_find};
        let mut rng: u64 = 0x853C49E6748FEA9B;
        let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };

        for d in [3usize, 5, 7, 9] {
            let code = crate::surface_code::RotatedSurfaceCode::new(d);
            let graph = code.build_syndrome_graph(d, true);
            let none = vec![false; graph.edges.len()];
            let weigh = |edges: &[usize]| edges.len();

            for rate in [0.02f64, 0.05, 0.10] {
                for _ in 0..40 {
                    let mut defects = vec![false; graph.num_nodes];
                    let mut count = 0;
                    for v in defects.iter_mut() {
                        if ((next() >> 11) as f64 / 9007199254740992.0) < rate {
                            *v = true;
                            count += 1;
                        }
                    }
                    if count == 0 { continue; }
                    let mw = weigh(&decode_mwpm(&graph, &defects, &none));
                    let gr = weigh(&decode_greedy(&graph, &defects, &none));
                    let uf = weigh(&decode_union_find(&graph, &defects, &none));
                    assert!(
                        mw <= gr && mw <= uf,
                        "d={d} rate={rate} defects={count}: mwpm={mw} greedy={gr} union-find={uf}"
                    );
                }
            }
        }
    }

    /// The Pauli frame must agree with the tableau, fault for fault.
    ///
    /// The XZZX circuit is simulated on the frame rather than the tableau, which
    /// is exact in theory for a Clifford circuit under Pauli noise — but only if
    /// the propagation rules are right. A wrong CZ rule, or an H that fails to
    /// swap, would give a self-consistent frame that quietly disagrees with the
    /// physics. So every single fault is injected into both and the detectors
    /// they fire are compared directly.
    #[test]
    fn xzzx_frame_propagation_agrees_with_the_tableau() {
        use crate::circuit_model::*;
        for d in [3usize, 5] {
            let code = crate::surface_code::XZZXSurfaceCode::new(d);
            let layout = code.circuit_layout();
            let program = &layout.program;
            let ns = code.stabilizers.len();
            let nq = code.data_qubits.len() + ns;
            const ROUNDS: usize = 3;
            let fault_round = 2usize;

            let run_tableau = |inject: Option<(usize, u8)>| -> Vec<Vec<u8>> {
                let mut sim = StabilizerSimulator::with_seed(nq, 20250818);
                let mut all = Vec::new();
                for r in 0..ROUNDS {
                    let mut out = vec![0u8; ns];
                    for (i, &op) in program.iter().enumerate() {
                        if r == fault_round {
                            if let Some((at, pauli)) = inject {
                                if at == i {
                                    if let Op::Noise(q) = op {
                                        if pauli & 1 != 0 { sim.apply_x(q); }
                                        if pauli & 2 != 0 { sim.apply_z(q); }
                                    }
                                }
                            }
                        }
                        match op {
                            Op::Reset(q) => { if sim.measure_z(q) == 1 { sim.apply_x(q); } }
                            Op::H(q) => sim.apply_h(q),
                            Op::Cnot(c, t) => sim.apply_cnot(c, t),
                            Op::Cz(a, b) => { sim.apply_h(b); sim.apply_cnot(a, b); sim.apply_h(b); }
                            Op::Measure(q, _, idx) => out[idx] = sim.measure_z(q),
                            Op::Noise(_) => {}
                        }
                    }
                    all.push(out);
                }
                all
            };

            let clean = run_tableau(None);
            let mut checked = 0usize;
            for (fault, _) in layout.fault_locations() {
                let (at, pauli) = match fault {
                    Fault::Gate(at, pauli) => (at, pauli),
                    Fault::Readout(_) => continue, // classical, nothing to propagate
                };
                let dirty = run_tableau(Some((at, pauli)));
                // What the tableau says this fault did to the readings.
                let tableau: Vec<bool> = (0..ns)
                    .map(|s| (dirty[fault_round][s] ^ clean[fault_round][s]) == 1)
                    .collect();
                // What the frame says.
                let effect = layout.propagate(fault, fault_round, ROUNDS);
                let frame: Vec<bool> = (0..ns)
                    .map(|s| effect.flips_z_stab[fault_round * ns + s])
                    .collect();
                assert_eq!(
                    tableau, frame,
                    "d={d}: frame and tableau disagree for {:?} at op {at}", pauli
                );
                checked += 1;
            }
            assert!(checked > 100, "d={d}: only {checked} faults compared");
            println!("d={d}: frame matches tableau on {checked} faults");
        }
    }

    /// Every fault must fire at most two detectors, or the graph cannot express
    /// it and the matcher will explain it with unrelated edges.
    #[test]
    fn xzzx_detector_model_is_graphlike() {
        for d in [3usize, 5, 7] {
            let code = crate::surface_code::XZZXSurfaceCode::new(d);
            let layout = code.circuit_layout();
            let (buckets, edges) = crate::circuit_model::stats_combined(&layout, d);
            assert!(edges > 0, "d={d}: model has no edges");
            assert_eq!(buckets[3], 0, "d={d}: {} faults fire 3 detectors", buckets[3]);
            assert_eq!(buckets[4], 0, "d={d}: {} faults fire 4+ detectors", buckets[4]);
        }
    }

    #[test]
    #[ignore]
    fn xzzx_search_schedules() {
        use crate::circuit_model::*;
        const DIRS: [(i32, i32, bool); 4] =
            [(-1, -1, true), (1, -1, false), (-1, 1, false), (1, 1, true)];
        let mut perms: Vec<[usize; 4]> = Vec::new();
        for a in 0..4 { for b in 0..4 { for c in 0..4 { for e in 0..4 {
            let v = [a, b, c, e];
            let mut seen = [false; 4];
            if v.iter().all(|&i| { let n = !seen[i]; seen[i] = true; n }) { perms.push(v); }
        }}}}
        let ord = |p: &[usize; 4]| [DIRS[p[0]], DIRS[p[1]], DIRS[p[2]], DIRS[p[3]]];

        let check = |d: usize, oa: &[(i32,i32,bool);4], ob: &[(i32,i32,bool);4]| -> (bool, usize, usize) {
            let code = crate::surface_code::XZZXSurfaceCode::new(d);
            let program = code.round_program_ordered(oa, ob);
            let ns = code.stabilizers.len();
            let mut sim = crate::simulator::StabilizerSimulator::with_seed(
                code.data_qubits.len() + ns, 7);
            let mut rr: Vec<Vec<u8>> = Vec::new();
            for _ in 0..3 {
                let mut out = vec![0u8; ns];
                for &op in &program {
                    match op {
                        Op::Reset(q) => { if sim.measure_z(q) == 1 { sim.apply_x(q); } }
                        Op::H(q) => sim.apply_h(q),
                        Op::Cnot(c, t) => sim.apply_cnot(c, t),
                        Op::Cz(a, b) => { sim.apply_h(b); sim.apply_cnot(a, b); sim.apply_h(b); }
                        Op::Measure(q, _, i) => out[i] = sim.measure_z(q),
                        Op::Noise(_) => {}
                    }
                }
                rr.push(out);
            }
            let commutes = rr[1] == rr[2];
            let layout = CircuitLayout {
                program,
                num_qubits: code.data_qubits.len() + ns,
                num_data: code.data_qubits.len(),
                num_x_stabs: 0,
                num_z_stabs: ns,
            };
            let model = build_combined(&layout, d);
            let (t, f) = single_fault_failures_combined(
                &layout, &model, d, 0, &|x, z| code.logical.is_logical(x, z));
            (commutes, f, t)
        };

        let mut passes = Vec::new();
        for pa in &perms {
            for pb in &perms {
                let (oa, ob) = (ord(pa), ord(pb));
                let (c3, f3, _) = check(3, &oa, &ob);
                if !c3 || f3 > 0 { continue; }
                let (c5, f5, _) = check(5, &oa, &ob);
                if c5 && f5 == 0 {
                    println!("PASS  A={:?}  B={:?}", pa, pb);
                    passes.push((*pa, *pb));
                }
            }
        }
        println!("total passing: {}", passes.len());
    }

    /// A distance-d code must survive any single fault. Deterministic and
    /// complete — the same bar the rotated code is held to.
    #[test]
    fn xzzx_survives_every_single_circuit_fault() {
        for d in [3usize, 5, 7] {
            let code = crate::surface_code::XZZXSurfaceCode::new(d);
            let layout = code.circuit_layout();
            let model = crate::circuit_model::build_combined(&layout, d);
            for decoder in [0usize, 2] {
                let (tested, failures) = crate::circuit_model::single_fault_failures_combined(
                    &layout, &model, d, decoder,
                    &|rx, rz| code.logical.is_logical(rx, rz),
                );
                assert!(tested > 0, "d={d}: nothing tested");
                println!("XZZX d={d} decoder={decoder}: {failures}/{tested}");
                assert_eq!(failures, 0, "d={d} decoder={decoder}: {failures}/{tested} faults uncorrected");
            }
        }
    }

    #[test]
    fn test_bell_state() {
        for _ in 0..100 {
            let mut sim = StabilizerSimulator::new(2);
            sim.apply_h(0);
            sim.apply_cnot(0, 1);

            let m0 = sim.measure_z(0);
            let m1 = sim.measure_z(1);

            // In a Bell state, measuring both qubits in Z basis must yield identical outcomes (00 or 11)
            assert_eq!(m0, m1);
        }
    }

    #[test]
    fn test_ghz_state() {
        for _ in 0..100 {
            let mut sim = StabilizerSimulator::new(3);
            sim.apply_h(0);
            sim.apply_cnot(0, 1);
            sim.apply_cnot(1, 2);

            let m0 = sim.measure_z(0);
            let m1 = sim.measure_z(1);
            let m2 = sim.measure_z(2);

            // GHZ state must result in either 000 or 111
            assert_eq!(m0, m1);
            assert_eq!(m1, m2);
        }
    }

    #[test]
    fn test_teleportation_x_basis() {
        for _ in 0..100 {
            let mut sim = StabilizerSimulator::new(3);
            // Qubit 0: message to teleport. Set to |+> state
            sim.apply_h(0);

            // Qubits 1 & 2: EPR pair
            sim.apply_h(1);
            sim.apply_cnot(1, 2);

            // Bell measurement on 0 & 1
            sim.apply_cnot(0, 1);
            sim.apply_h(0);

            let m1 = sim.measure_z(1);
            let m0 = sim.measure_z(0);

            // Active feedback on 2
            if m1 == 1 {
                sim.apply_x(2);
            }
            if m0 == 1 {
                sim.apply_z(2);
            }

            // Qubit 2 should now be in the |+> state.
            // Measuring it in X basis must always yield 0 (which means +1 eigenstate).
            let m2 = sim.measure_x(2);
            assert_eq!(m2, 0);
        }
    }

    #[test]
    fn test_teleportation_y_basis() {
        for _ in 0..100 {
            let mut sim = StabilizerSimulator::new(3);
            // Qubit 0: message to teleport. Set to |i> state (Y-eigenstate)
            sim.apply_h(0);
            sim.apply_s(0);

            // Qubits 1 & 2: EPR pair
            sim.apply_h(1);
            sim.apply_cnot(1, 2);

            // Bell measurement on 0 & 1
            sim.apply_cnot(0, 1);
            sim.apply_h(0);

            let m1 = sim.measure_z(1);
            let m0 = sim.measure_z(0);

            // Active feedback on 2
            if m1 == 1 {
                sim.apply_x(2);
            }
            if m0 == 1 {
                sim.apply_z(2);
            }

            // Qubit 2 should now be in the |i> state.
            // Measuring it in Y basis must always yield 0 (+1 eigenstate).
            let m2 = sim.measure_y(2);
            assert_eq!(m2, 0);
        }
    }

    #[test]
    fn test_single_qubit_gates() {
        let mut sim = StabilizerSimulator::new(1);
        
        // Z gate on |0> does nothing (phase remains +1)
        sim.apply_z(0);
        assert_eq!(sim.measure_z(0), 0);

        // X gate flips |0> to |1>
        sim.apply_x(0);
        assert_eq!(sim.measure_z(0), 1);

        // Y gate flips |1> to |0> (ignoring global phase)
        sim.apply_y(0);
        assert_eq!(sim.measure_z(0), 0);
    }

    #[test]
    fn test_surface_code_zero_noise() {
        let code = crate::surface_code::RotatedSurfaceCode::new(3);
        for _ in 0..10 {
            let logical_err = code.simulate_phenomenological_noise(3, 0.0, 1.0, 0, 0.0, 0);
            assert_eq!(logical_err, 0);
        }
    }

    #[test]
    fn test_surface_code_low_noise() {
        let code = crate::surface_code::RotatedSurfaceCode::new(3);
        let mut error_count = 0;
        let num_runs = 500;
        for _ in 0..num_runs {
            if code.simulate_phenomenological_noise(3, 0.005, 1.0, 0, 0.0, 0) != 0 {
                error_count += 1;
            }
        }
        let error_rate = (error_count as f64) / (num_runs as f64);
        println!("d=3, p=0.005: logical error rate = {}", error_rate);
        assert!(error_rate < 0.12, "Logical error rate {} too high for p=0.005", error_rate);
    }

    #[test]
    fn test_xzzx_commutation() {
        let code = crate::surface_code::XZZXSurfaceCode::new(3);
        let num_stabs = code.stabilizers.len();
        let num_data = code.data_qubits.len();

        // Build symplectic representation of each stabilizer
        let mut x_ops = vec![vec![false; num_data]; num_stabs];
        let mut z_ops = vec![vec![false; num_data]; num_stabs];

        for s_idx in 0..num_stabs {
            let (sx, sy) = code.stabilizers[s_idx];
            // NW: X
            if let Some(q) = code.get_neighbor_idx(sx as i32 - 1, sy as i32 - 1) {
                x_ops[s_idx][q] = true;
            }
            // SE: X
            if let Some(q) = code.get_neighbor_idx(sx as i32 + 1, sy as i32 + 1) {
                x_ops[s_idx][q] = true;
            }
            // NE: Z
            if let Some(q) = code.get_neighbor_idx(sx as i32 + 1, sy as i32 - 1) {
                z_ops[s_idx][q] = true;
            }
            // SW: Z
            if let Some(q) = code.get_neighbor_idx(sx as i32 - 1, sy as i32 + 1) {
                z_ops[s_idx][q] = true;
            }
        }

        // Check that all pairs of stabilizers commute
        for i in 0..num_stabs {
            for j in 0..num_stabs {
                let mut inner_product = false;
                for q in 0..num_data {
                    if (x_ops[i][q] && z_ops[j][q]) ^ (z_ops[i][q] && x_ops[j][q]) {
                        inner_product ^= true;
                    }
                }
                assert!(!inner_product, "Stabilizers {} and {} do not commute!", i, j);
            }
        }
    }

    #[test]
    fn test_greedy_decoder() {
        let code = crate::surface_code::RotatedSurfaceCode::new(3);
        let graph_z = code.build_syndrome_graph(1, true);

        // Inject a single X error on data qubit 0
        let mut defects = vec![false; graph_z.num_nodes];
        // Data qubit 0 is connected to Z-stabilizer at (2,2), which is index 1.
        defects[1] = true; // Z-stabilizer (2,2) should trigger

        let erased = vec![false; graph_z.edges.len()];
        let correction = crate::decoder::decode_greedy(&graph_z, &defects, &erased);
        // Greedy matching should find the single error and match it to the nearest boundary.
        // It should return 1 correction edge.
        assert_eq!(correction.len(), 1);
        let corrected_qubit = graph_z.edge_to_qubit[correction[0]].unwrap();
        assert_eq!(corrected_qubit, 0); // Should correct qubit 0
    }

    #[test]
    fn test_mwpm_decoder() {
        let code = crate::surface_code::RotatedSurfaceCode::new(3);
        let graph_z = code.build_syndrome_graph(1, true);

        // Inject a single X error on data qubit 0
        let mut defects = vec![false; graph_z.num_nodes];
        defects[1] = true;

        let erased = vec![false; graph_z.edges.len()];
        let correction = crate::decoder::decode_mwpm(&graph_z, &defects, &erased);
        assert_eq!(correction.len(), 1);
        let corrected_qubit = graph_z.edge_to_qubit[correction[0]].unwrap();
        assert_eq!(corrected_qubit, 0);
    }

    #[test]
    fn test_dijkstra_erasure() {
        let code = crate::surface_code::RotatedSurfaceCode::new(3);
        let graph_z = code.build_syndrome_graph(1, true);
        let mut erased_edges = vec![false; graph_z.edges.len()];
        // Erase the edge representing data qubit 0 (which is index 0)
        erased_edges[0] = true;
        let mut defects = vec![false; graph_z.num_nodes];
        defects[1] = true;
        let correction = crate::decoder::decode_mwpm(&graph_z, &defects, &erased_edges);
        assert!(!correction.is_empty());
    }

    #[test]
    fn test_circuit_level_noise_zero_noise() {
        let code_rotated = crate::surface_code::RotatedSurfaceCode::new(3);
        let layout = code_rotated.circuit_layout();
        let model = crate::circuit_model::build(&layout, 2);
        let failed_rot =
            code_rotated.simulate_circuit_noise_with_model(&model, 2, 0.0, 1.0, "zero", 0, 0.0, 0);
        assert_eq!(failed_rot, 0);

        let code_xzzx = crate::surface_code::XZZXSurfaceCode::new(3);
        let xzzx_layout = code_xzzx.circuit_layout();
        let xzzx_model = crate::circuit_model::build_combined(&xzzx_layout, 2);
        let failed_xzzx =
            code_xzzx.simulate_circuit_noise_with_model(&xzzx_model, 2, 0.0, 1.0, 0, 0.0, 0);
        assert_eq!(failed_xzzx, 0);
    }
}

/// Noise locations in one extraction round, for the two codes. Diagnostic.
#[cfg(not(feature = "python"))]
#[no_mangle]
pub extern "C" fn wasm_noise_slots(d: usize, code_type: usize) -> usize {
    if code_type == 0 {
        surface_code::RotatedSurfaceCode::new(d).circuit_layout().noise_slots()
    } else {
        surface_code::XZZXSurfaceCode::new(d).circuit_layout().noise_slots()
    }
}

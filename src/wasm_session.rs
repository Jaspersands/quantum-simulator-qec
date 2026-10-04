//! The WebAssembly surface of the explainer's first sections: a session holding one code patch,
//! its errors and its decoder's correction, driven through C-ABI exports by js/engine.js. Not part
//! of the crate's API.

#![allow(unused_imports)]
use crate::*;

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
    // Only the per-code paths estimate a channel; any other mode is refused with
    // NaN rather than counted as a run of perfect shots.
    if noise_mode > 2 {
        unsafe {
            let out = std::ptr::addr_of_mut!(FIDELITY_RESULTS);
            (*out) = [f64::NAN; 3];
            return out as *const f64;
        }
    }

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
    // Noise mode 3 is SD6, which only the general path models.
    #[cfg(not(feature = "python"))]
    if noise_mode == 3 {
        return wasm_xc::wasm_xc_run(code_type, d, num_rounds, 1, p, bias, num_runs, 1);
    }
    // An unknown mode is refused with NaN, never reported as zero failures.
    if noise_mode > 3 {
        return f64::NAN;
    }

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

/// Largest matching this export accepts; matches `blossom::MAX_VERTICES`.
#[cfg(not(feature = "python"))]
pub const MATCH_MAX: usize = 512;

#[cfg(not(feature = "python"))]
static mut MATCH_COST: [i64; MATCH_MAX * MATCH_MAX] = [0; MATCH_MAX * MATCH_MAX];

#[cfg(not(feature = "python"))]
static mut MATCH_OUT: [u32; MATCH_MAX] = [0; MATCH_MAX];

/// Row-major cost buffer for `wasm_match_solve`. Write `n*n` weights here first.
///
/// Exposed so the independent cross-check in `tools/` can use the same exact
/// matcher the decoder does. That is deliberate: the point of that harness is to
/// test the *physics* against a second implementation, and it shares no lattice,
/// noise, graph or scoring with the engine. Its decoder being weaker was a
/// confound rather than a control — a poor matcher shifts the threshold, and at
/// these sizes can distort the exponent too. Sharing a matcher that has been
/// checked against brute force on 1,500 instances removes that confound without
/// touching anything the comparison is about.
#[cfg(not(feature = "python"))]
#[no_mangle]
pub extern "C" fn wasm_match_cost_ptr() -> *mut i64 {
    std::ptr::addr_of_mut!(MATCH_COST) as *mut i64
}

/// Solve the matching in the cost buffer. Returns a pointer to `n` partners, or
/// all `u32::MAX` if the matcher declined.
#[cfg(not(feature = "python"))]
#[no_mangle]
pub extern "C" fn wasm_match_solve(n: usize) -> *const u32 {
    if n > MATCH_MAX {
        unsafe {
            let out = std::ptr::addr_of_mut!(MATCH_OUT) as *mut u32;
            for i in 0..MATCH_MAX {
                *out.add(i) = u32::MAX;
            }
            return out as *const u32;
        }
    }
    let cost: Vec<Vec<i64>> = unsafe {
        let base = std::ptr::addr_of!(MATCH_COST) as *const i64;
        (0..n)
            .map(|i| (0..n).map(|j| *base.add(i * n + j)).collect())
            .collect()
    };
    let mate = blossom::min_weight_perfect_matching(n, &cost);
    unsafe {
        let out = std::ptr::addr_of_mut!(MATCH_OUT) as *mut u32;
        match mate {
            Some(m) => {
                for (i, &partner) in m.iter().enumerate() {
                    *out.add(i) = partner as u32;
                }
            }
            None => {
                for i in 0..n {
                    *out.add(i) = u32::MAX;
                }
            }
        }
        out as *const u32
    }
}

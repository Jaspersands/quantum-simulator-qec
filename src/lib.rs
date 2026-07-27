pub mod tableau;
pub mod simulator;
pub mod decoder;
pub mod surface_code;

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
        self.code.simulate_phenomenological_noise(num_rounds, p, bias.unwrap_or(1.0), decoder_type.unwrap_or(0), 0.0, 0)
    }

    #[pyo3(signature = (p, bias=None, decoder_type=None))]
    fn simulate_data_noise(&self, p: f64, bias: Option<f64>, decoder_type: Option<usize>) -> bool {
        self.code.simulate_data_noise(p, bias.unwrap_or(1.0), decoder_type.unwrap_or(0), 0.0, 0)
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
pub extern "C" fn wasm_get_data_qubit_coord(ptr: *mut WasmSession, idx: usize, out_xy: *mut usize) {
    let session = unsafe { &*ptr };
    if session.code_type == 0 {
        let code = surface_code::RotatedSurfaceCode::new(session.d);
        if idx < code.data_qubits.len() {
            let (x, y) = code.data_qubits[idx];
            unsafe {
                *out_xy = x;
                *out_xy.add(1) = y;
            }
        }
    } else {
        let code = surface_code::XZZXSurfaceCode::new(session.d);
        if idx < code.data_qubits.len() {
            let (x, y) = code.data_qubits[idx];
            unsafe {
                *out_xy = x;
                *out_xy.add(1) = y;
            }
        }
    }
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
pub extern "C" fn wasm_get_stabilizer_coord(ptr: *mut WasmSession, idx: usize, out_xy: *mut usize) {
    let session = unsafe { &*ptr };
    if session.code_type == 0 {
        let code = surface_code::RotatedSurfaceCode::new(session.d);
        let num_x = code.x_stabilizers.len();
        if idx < num_x {
            let (x, y) = code.x_stabilizers[idx];
            unsafe {
                *out_xy = x;
                *out_xy.add(1) = y;
            }
        } else if idx < num_x + code.z_stabilizers.len() {
            let (x, y) = code.z_stabilizers[idx - num_x];
            unsafe {
                *out_xy = x;
                *out_xy.add(1) = y;
            }
        }
    } else {
        let code = surface_code::XZZXSurfaceCode::new(session.d);
        if idx < code.stabilizers.len() {
            let (x, y) = code.stabilizers[idx];
            unsafe {
                *out_xy = x;
                *out_xy.add(1) = y;
            }
        }
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
            let q_idx = 0 + d * y_idx;
            if accumulated_x[q_idx] {
                logical_x ^= true;
            }
        }

        let mut logical_z = false;
        for x_idx in 0..d {
            let q_idx = x_idx + d * 0;
            if accumulated_z[q_idx] {
                logical_z ^= true;
            }
        }

        if logical_x || logical_z { 1 } else { 0 }
    } else {
        let code = surface_code::XZZXSurfaceCode::new(d);
        let num_stabs = code.stabilizers.len();

        let graph_z = code.build_syndrome_graph(num_rounds, true);
        let mut defects_z = vec![false; graph_z.num_nodes];
        for t in 0..num_rounds {
            for s_idx in 0..num_stabs {
                let outcome = session.syndrome[s_idx + t * num_stabs] == 1;
                let prev_outcome = if t == 0 { false } else { session.syndrome[s_idx + (t - 1) * num_stabs] == 1 };
                defects_z[s_idx + t * num_stabs] = outcome ^ prev_outcome;
            }
        }
        let mut erased_edges_z = vec![false; graph_z.edges.len()];
        for edge_idx in 0..graph_z.edges.len() {
            if let Some(q_idx) = graph_z.edge_to_qubit[edge_idx] {
                let u = graph_z.edges[edge_idx].u;
                let t_layer = u / num_stabs;
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
                let t_layer = u / num_stabs;
                if t_layer < num_rounds {
                    session.correction_x[q_idx + t_layer * num_data] ^= 1;
                }
            }
        }

        let graph_x = code.build_syndrome_graph(num_rounds, false);
        let defects_x = defects_z.clone();
        let mut erased_edges_x = vec![false; graph_x.edges.len()];
        for edge_idx in 0..graph_x.edges.len() {
            if let Some(q_idx) = graph_x.edge_to_qubit[edge_idx] {
                let u = graph_x.edges[edge_idx].u;
                let t_layer = u / num_stabs;
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
                let t_layer = u / num_stabs;
                if t_layer < num_rounds {
                    session.correction_z[q_idx + t_layer * num_data] ^= 1;
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

        let mut logical_err_1 = false;
        for x_idx in 0..d {
            let q = x_idx + d * 0;
            let err = if x_idx % 2 == 0 { accumulated_z[q] } else { accumulated_x[q] };
            if err {
                logical_err_1 ^= true;
            }
        }

        let mut logical_err_2 = false;
        for y_idx in 0..d {
            let q = 0 + d * y_idx;
            let err = if y_idx % 2 == 0 { accumulated_x[q] } else { accumulated_z[q] };
            if err {
                logical_err_2 ^= true;
            }
        }

        if logical_err_1 || logical_err_2 { 1 } else { 0 }
    }
}

static mut FIDELITY_RESULTS: [f64; 3] = [0.0; 3];

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
    let mut failures_x = 0;
    let mut failures_y = 0;
    let mut failures_z = 0;

    if code_type == 0 {
        let code = surface_code::RotatedSurfaceCode::new(d);
        for _ in 0..runs {
            let failed_x = match noise_mode {
                0 => code.simulate_data_noise(p, bias, decoder_type, erasure_rate, correlated_noise),
                1 => code.simulate_phenomenological_noise(num_rounds, p, bias, decoder_type, erasure_rate, correlated_noise),
                2 => code.simulate_circuit_noise(num_rounds, p, bias, "plus", decoder_type, erasure_rate, correlated_noise),
                _ => false,
            };
            if failed_x {
                failures_x += 1;
            }

            let failed_y = match noise_mode {
                0 => code.simulate_data_noise(p, bias, decoder_type, erasure_rate, correlated_noise),
                1 => code.simulate_phenomenological_noise(num_rounds, p, bias, decoder_type, erasure_rate, correlated_noise),
                2 => code.simulate_circuit_noise(num_rounds, p, bias, "y", decoder_type, erasure_rate, correlated_noise),
                _ => false,
            };
            if failed_y {
                failures_y += 1;
            }

            let failed_z = match noise_mode {
                0 => code.simulate_data_noise(p, bias, decoder_type, erasure_rate, correlated_noise),
                1 => code.simulate_phenomenological_noise(num_rounds, p, bias, decoder_type, erasure_rate, correlated_noise),
                2 => code.simulate_circuit_noise(num_rounds, p, bias, "zero", decoder_type, erasure_rate, correlated_noise),
                _ => false,
            };
            if failed_z {
                failures_z += 1;
            }
        }
    } else {
        let code = surface_code::XZZXSurfaceCode::new(d);
        for _ in 0..runs {
            let failed_x = match noise_mode {
                0 => code.simulate_data_noise(p, bias, decoder_type, erasure_rate, correlated_noise),
                1 => code.simulate_phenomenological_noise(num_rounds, p, bias, decoder_type, erasure_rate, correlated_noise),
                2 => code.simulate_circuit_noise(num_rounds, p, bias, "plus", decoder_type, erasure_rate, correlated_noise),
                _ => false,
            };
            if failed_x {
                failures_x += 1;
            }

            let failed_y = match noise_mode {
                0 => code.simulate_data_noise(p, bias, decoder_type, erasure_rate, correlated_noise),
                1 => code.simulate_phenomenological_noise(num_rounds, p, bias, decoder_type, erasure_rate, correlated_noise),
                2 => code.simulate_circuit_noise(num_rounds, p, bias, "y", decoder_type, erasure_rate, correlated_noise),
                _ => false,
            };
            if failed_y {
                failures_y += 1;
            }

            let failed_z = match noise_mode {
                0 => code.simulate_data_noise(p, bias, decoder_type, erasure_rate, correlated_noise),
                1 => code.simulate_phenomenological_noise(num_rounds, p, bias, decoder_type, erasure_rate, correlated_noise),
                2 => code.simulate_circuit_noise(num_rounds, p, bias, "zero", decoder_type, erasure_rate, correlated_noise),
                _ => false,
            };
            if failed_z {
                failures_z += 1;
            }
        }
    }

    let p_x = (failures_x as f64) / (runs as f64);
    let p_y = (failures_y as f64) / (runs as f64);
    let p_z = (failures_z as f64) / (runs as f64);

    unsafe {
        FIDELITY_RESULTS[0] = 1.0 - 2.0 * p_x;
        FIDELITY_RESULTS[1] = 1.0 - 2.0 * p_y;
        FIDELITY_RESULTS[2] = 1.0 - 2.0 * p_z;
        std::ptr::addr_of!(FIDELITY_RESULTS) as *const f64
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
        for _ in 0..num_runs {
            let failed = match noise_mode {
                0 => code.simulate_data_noise(p, bias, decoder_type, erasure_rate, correlated_noise),
                1 => code.simulate_phenomenological_noise(num_rounds, p, bias, decoder_type, erasure_rate, correlated_noise),
                2 => code.simulate_circuit_noise(num_rounds, p, bias, "zero", decoder_type, erasure_rate, correlated_noise),
                _ => false,
            };
            if failed {
                failures += 1;
            }
        }
    } else {
        let code = surface_code::XZZXSurfaceCode::new(d);
        for _ in 0..num_runs {
            let failed = match noise_mode {
                0 => code.simulate_data_noise(p, bias, decoder_type, erasure_rate, correlated_noise),
                1 => code.simulate_phenomenological_noise(num_rounds, p, bias, decoder_type, erasure_rate, correlated_noise),
                2 => code.simulate_circuit_noise(num_rounds, p, bias, "zero", decoder_type, erasure_rate, correlated_noise),
                _ => false,
            };
            if failed {
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
            assert!(!logical_err);
        }
    }

    #[test]
    fn test_surface_code_low_noise() {
        let code = crate::surface_code::RotatedSurfaceCode::new(3);
        let mut error_count = 0;
        let num_runs = 500;
        for _ in 0..num_runs {
            if code.simulate_phenomenological_noise(3, 0.005, 1.0, 0, 0.0, 0) {
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
        let failed_rot = code_rotated.simulate_circuit_noise(2, 0.0, 1.0, "zero", 0, 0.0, 0);
        assert!(!failed_rot);

        let code_xzzx = crate::surface_code::XZZXSurfaceCode::new(3);
        let failed_xzzx = code_xzzx.simulate_circuit_noise(2, 0.0, 1.0, "zero", 0, 0.0, 0);
        assert!(!failed_xzzx);
    }
}

use crate::decoder::{Edge, SyndromeGraph};

pub struct Xorshift {
    state: u64,
}

impl Xorshift {
    pub fn new(seed: u64) -> Self {
        let mut s = seed;
        if s == 0 {
            s = 0xdeadbeef;
        }
        Xorshift { state: s }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub fn next_f64(&mut self) -> f64 {
        let val = self.next_u64();
        (val as f64) / (u64::MAX as f64)
    }
}

fn sample_biased_error(p: f64, eta: f64, rng: &mut Xorshift) -> (bool, bool) {
    if rng.next_f64() < p {
        let rand_val = rng.next_f64();
        let p_z_ratio = eta / (eta + 1.0);
        let p_x_ratio = 1.0 / (2.0 * (eta + 1.0));
        if rand_val < p_z_ratio {
            (false, true) // Z error
        } else if rand_val < p_z_ratio + p_x_ratio {
            (true, false) // X error
        } else {
            (true, true)  // Y error
        }
    } else {
        (false, false)
    }
}

fn inject_single_qubit_noise(sim: &mut crate::simulator::StabilizerSimulator, qubit: usize, p: f64, bias: f64, rng: &mut Xorshift) {
    if rng.next_f64() < p {
        let rand_val = rng.next_f64();
        let p_z_ratio = bias / (bias + 1.0);
        let p_x_ratio = 1.0 / (2.0 * (bias + 1.0));
        if rand_val < p_z_ratio {
            sim.apply_z(qubit);
        } else if rand_val < p_z_ratio + p_x_ratio {
            sim.apply_x(qubit);
        } else {
            sim.apply_y(qubit);
        }
    }
}

fn inject_two_qubit_noise(sim: &mut crate::simulator::StabilizerSimulator, q1: usize, q2: usize, p: f64, bias: f64, rng: &mut Xorshift) {
    inject_single_qubit_noise(sim, q1, p / 2.0, bias, rng);
    inject_single_qubit_noise(sim, q2, p / 2.0, bias, rng);
}

fn decode_by_type(graph: &SyndromeGraph, defects: &[bool], decoder_type: usize) -> Vec<usize> {
    match decoder_type {
        1 => crate::decoder::decode_greedy(graph, defects),
        2 => crate::decoder::decode_mwpm(graph, defects),
        _ => crate::decoder::decode_union_find(graph, defects),
    }
}

pub struct RotatedSurfaceCode {
    pub d: usize,
    pub data_qubits: Vec<(usize, usize)>,
    pub x_stabilizers: Vec<(usize, usize)>,
    pub z_stabilizers: Vec<(usize, usize)>,
}

impl RotatedSurfaceCode {
    pub fn new(d: usize) -> Self {
        let mut data_qubits = Vec::new();
        for y in (1..(2 * d)).step_by(2) {
            for x in (1..(2 * d)).step_by(2) {
                data_qubits.push((x, y));
            }
        }

        let mut x_stabilizers = Vec::new();
        let mut z_stabilizers = Vec::new();

        // Construct Z-stabilizers
        // Z-type boundary checks are on top/bottom, so we restrict x to [2, 2d-2] to avoid left/right boundary Z-checks.
        for y in (0..=(2 * d)).step_by(2) {
            for x in (2..(2 * d)).step_by(2) {
                if ((x + y) / 2) % 2 == 0 {
                    z_stabilizers.push((x, y));
                }
            }
        }

        // Construct X-stabilizers
        // X-type boundary checks are on left/right, so we restrict y to [2, 2d-2] to avoid top/bottom boundary X-checks.
        for y in (2..(2 * d)).step_by(2) {
            for x in (0..=(2 * d)).step_by(2) {
                if ((x + y) / 2) % 2 == 1 {
                    x_stabilizers.push((x, y));
                }
            }
        }

        RotatedSurfaceCode {
            d,
            data_qubits,
            x_stabilizers,
            z_stabilizers,
        }
    }

    pub fn get_neighbors(&self, stab: &(usize, usize)) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let (sx, sy) = *stab;
        for &(dx, dy) in &[
            (sx as i32 - 1, sy as i32 - 1),
            (sx as i32 - 1, sy as i32 + 1),
            (sx as i32 + 1, sy as i32 - 1),
            (sx as i32 + 1, sy as i32 + 1),
        ] {
            if dx >= 1 && dx < (2 * self.d) as i32 && dy >= 1 && dy < (2 * self.d) as i32 {
                let x_idx = (dx - 1) as usize / 2;
                let y_idx = (dy - 1) as usize / 2;
                neighbors.push(x_idx + self.d * y_idx);
            }
        }
        neighbors
    }

    pub fn build_syndrome_graph(&self, num_rounds: usize, is_z_type: bool) -> SyndromeGraph {
        let stabilizers = if is_z_type {
            &self.z_stabilizers
        } else {
            &self.x_stabilizers
        };
        let num_stabs = stabilizers.len();
        let num_nodes = num_stabs * num_rounds;

        let mut edges = Vec::new();
        let mut edge_to_qubit = Vec::new();
        let mut edge_id = 0;

        // 1. Spatial Edges (errors on data qubits at round t)
        for t in 0..num_rounds {
            for q_idx in 0..self.data_qubits.len() {
                let q_coord = self.data_qubits[q_idx];
                // Find stabilizers connected to this data qubit
                let mut connected_stabs = Vec::new();
                for (s_idx, stab) in stabilizers.iter().enumerate() {
                    let (sx, sy) = *stab;
                    let (qx, qy) = q_coord;
                    if (sx as i32 - qx as i32).abs() == 1 && (sy as i32 - qy as i32).abs() == 1 {
                        connected_stabs.push(s_idx);
                    }
                }

                if connected_stabs.len() == 2 {
                    let u = connected_stabs[0] + t * num_stabs;
                    let v = connected_stabs[1] + t * num_stabs;
                    edges.push(Edge { u, v, id: edge_id });
                    edge_to_qubit.push(Some(q_idx));
                    edge_id += 1;
                } else if connected_stabs.len() == 1 {
                    let u = connected_stabs[0] + t * num_stabs;
                    let v = num_nodes;
                    edges.push(Edge { u, v, id: edge_id });
                    edge_to_qubit.push(Some(q_idx));
                    edge_id += 1;
                }
            }
        }

        // 2. Temporal Edges (measurement errors on stabilizers)
        for t in 0..(num_rounds - 1) {
            for s_idx in 0..num_stabs {
                let u = s_idx + t * num_stabs;
                let v = s_idx + (t + 1) * num_stabs;
                edges.push(Edge { u, v, id: edge_id });
                edge_to_qubit.push(None);
                edge_id += 1;
            }
        }

        SyndromeGraph {
            num_nodes,
            edges,
            edge_to_qubit,
        }
    }

    /// Simulates the QEC cycle under phenomenological noise.
    /// Returns true if a logical error occurs, false if the code successfully corrected all errors.
    pub fn simulate_phenomenological_noise(&self, num_rounds: usize, p: f64, bias: f64, decoder_type: usize) -> bool {
        #[cfg(feature = "python")]
        let mut rng = Xorshift::new(rand::random());
        #[cfg(not(feature = "python"))]
        let mut rng = Xorshift::new(12345);

        let num_stabs_z = self.z_stabilizers.len();
        let num_stabs_x = self.x_stabilizers.len();
        let num_data = self.data_qubits.len();

        let mut physical_x = vec![false; num_data];
        let mut physical_z = vec![false; num_data];

        let mut measured_z = vec![vec![false; num_stabs_z]; num_rounds];
        let mut measured_x = vec![vec![false; num_stabs_x]; num_rounds];

        for t in 0..num_rounds {
            for q in 0..num_data {
                let (err_x, err_z) = sample_biased_error(p, bias, &mut rng);
                if err_x { physical_x[q] ^= true; }
                if err_z { physical_z[q] ^= true; }
            }

            for s_idx in 0..num_stabs_z {
                let neighbors = self.get_neighbors(&self.z_stabilizers[s_idx]);
                let mut parity = false;
                for &q in &neighbors {
                    if physical_x[q] {
                        parity ^= true;
                    }
                }
                if rng.next_f64() < p {
                    parity ^= true;
                }
                measured_z[t][s_idx] = parity;
            }

            for s_idx in 0..num_stabs_x {
                let neighbors = self.get_neighbors(&self.x_stabilizers[s_idx]);
                let mut parity = false;
                for &q in &neighbors {
                    if physical_z[q] {
                        parity ^= true;
                    }
                }
                if rng.next_f64() < p {
                    parity ^= true;
                }
                measured_x[t][s_idx] = parity;
            }
        }

        let graph_z = self.build_syndrome_graph(num_rounds, true);
        let mut defects_z = vec![false; graph_z.num_nodes];
        for t in 0..num_rounds {
            for s_idx in 0..num_stabs_z {
                let prev_outcome = if t == 0 { false } else { measured_z[t - 1][s_idx] };
                let diff = measured_z[t][s_idx] ^ prev_outcome;
                defects_z[s_idx + t * num_stabs_z] = diff;
            }
        }
        let correction_z_edges = decode_by_type(&graph_z, &defects_z, decoder_type);

        let mut correction_x_data = vec![false; num_data];
        for edge_idx in correction_z_edges {
            if let Some(q_idx) = graph_z.edge_to_qubit[edge_idx] {
                correction_x_data[q_idx] ^= true;
            }
        }

        let graph_x = self.build_syndrome_graph(num_rounds, false);
        let mut defects_x = vec![false; graph_x.num_nodes];
        for t in 0..num_rounds {
            for s_idx in 0..num_stabs_x {
                let prev_outcome = if t == 0 { false } else { measured_x[t - 1][s_idx] };
                let diff = measured_x[t][s_idx] ^ prev_outcome;
                defects_x[s_idx + t * num_stabs_x] = diff;
            }
        }
        let correction_x_edges = decode_by_type(&graph_x, &defects_x, decoder_type);

        let mut correction_z_data = vec![false; num_data];
        for edge_idx in correction_x_edges {
            if let Some(q_idx) = graph_x.edge_to_qubit[edge_idx] {
                correction_z_data[q_idx] ^= true;
            }
        }

        let mut residual_x = vec![false; num_data];
        let mut residual_z = vec![false; num_data];
        for q in 0..num_data {
            residual_x[q] = physical_x[q] ^ correction_x_data[q];
            residual_z[q] = physical_z[q] ^ correction_z_data[q];
        }

        // A logical X error occurred if the residual X errors cross the code from left to right.
        // We check this by checking the parity of residual X errors along a vertical line (x=1).
        let mut logical_x = false;
        for y_idx in 0..self.d {
            let q_idx = 0 + self.d * y_idx; // Qubits along x=1 (indices 0, d, 2d, ...)
            if residual_x[q_idx] {
                logical_x ^= true;
            }
        }

        // A logical Z error occurred if the residual Z errors cross the code from top to bottom.
        // We check this by checking the parity of residual Z errors along a horizontal line (y=1).
        let mut logical_z = false;
        for x_idx in 0..self.d {
            let q_idx = x_idx + self.d * 0; // Qubits along y=1 (indices 0, 1, 2, ..., d-1)
            if residual_z[q_idx] {
                logical_z ^= true;
            }
        }

        logical_x || logical_z
    }

    /// Simulates pure data noise with perfect stabilizer measurements.
    pub fn simulate_data_noise(&self, p: f64, bias: f64, decoder_type: usize) -> bool {
        #[cfg(feature = "python")]
        let mut rng = Xorshift::new(rand::random());
        #[cfg(not(feature = "python"))]
        let mut rng = Xorshift::new(12345);

        let num_data = self.data_qubits.len();
        let mut physical_x = vec![false; num_data];
        let mut physical_z = vec![false; num_data];
        
        for q in 0..num_data {
            let (err_x, err_z) = sample_biased_error(p, bias, &mut rng);
            if err_x { physical_x[q] = true; }
            if err_z { physical_z[q] = true; }
        }
        
        let num_stabs_z = self.z_stabilizers.len();
        let mut measured_z = vec![false; num_stabs_z];
        for s_idx in 0..num_stabs_z {
            let neighbors = self.get_neighbors(&self.z_stabilizers[s_idx]);
            let mut parity = false;
            for &q in &neighbors {
                if physical_x[q] {
                    parity ^= true;
                }
            }
            measured_z[s_idx] = parity;
        }

        let num_stabs_x = self.x_stabilizers.len();
        let mut measured_x = vec![false; num_stabs_x];
        for s_idx in 0..num_stabs_x {
            let neighbors = self.get_neighbors(&self.x_stabilizers[s_idx]);
            let mut parity = false;
            for &q in &neighbors {
                if physical_z[q] {
                    parity ^= true;
                }
            }
            measured_x[s_idx] = parity;
        }

        let graph_z = self.build_syndrome_graph(1, true);
        let correction_z_edges = decode_by_type(&graph_z, &measured_z, decoder_type);
        let mut correction_x_data = vec![false; num_data];
        for edge_idx in correction_z_edges {
            if let Some(q_idx) = graph_z.edge_to_qubit[edge_idx] {
                correction_x_data[q_idx] ^= true;
            }
        }

        let graph_x = self.build_syndrome_graph(1, false);
        let correction_x_edges = decode_by_type(&graph_x, &measured_x, decoder_type);
        let mut correction_z_data = vec![false; num_data];
        for edge_idx in correction_x_edges {
            if let Some(q_idx) = graph_x.edge_to_qubit[edge_idx] {
                correction_z_data[q_idx] ^= true;
            }
        }

        let mut residual_x = vec![false; num_data];
        let mut residual_z = vec![false; num_data];
        for q in 0..num_data {
            residual_x[q] = physical_x[q] ^ correction_x_data[q];
            residual_z[q] = physical_z[q] ^ correction_z_data[q];
        }

        let mut logical_x = false;
        for y_idx in 0..self.d {
            let q_idx = 0 + self.d * y_idx;
            if residual_x[q_idx] {
                logical_x ^= true;
            }
        }

        let mut logical_z = false;
        for x_idx in 0..self.d {
            let q_idx = x_idx + self.d * 0;
            if residual_z[q_idx] {
                logical_z ^= true;
            }
        }

        logical_x || logical_z
    }

    pub fn simulate_circuit_noise(&self, num_rounds: usize, p: f64, bias: f64, init_state: &str, decoder_type: usize) -> bool {
        let num_data = self.data_qubits.len();
        let num_stabs_x = self.x_stabilizers.len();
        let num_stabs_z = self.z_stabilizers.len();
        let num_stabs = num_stabs_x + num_stabs_z;
        let total_qubits = num_data + num_stabs;
        
        let mut sim = crate::simulator::StabilizerSimulator::new(total_qubits);
        #[cfg(feature = "python")]
        let mut rng = Xorshift::new(rand::random());
        #[cfg(not(feature = "python"))]
        let mut rng = Xorshift::new(54321);
 
        let get_x_stab_qubit = |j: usize| num_data + j;
        let get_z_stab_qubit = |k: usize| num_data + num_stabs_x + k;
 
        if init_state == "plus" {
            for i in 0..num_data {
                sim.apply_h(i);
                inject_single_qubit_noise(&mut sim, i, p, bias, &mut rng);
            }
        } else {
            for i in 0..num_data {
                inject_single_qubit_noise(&mut sim, i, p, bias, &mut rng);
            }
        }
 
        // Helper to run one round of syndrome extraction
        let get_neighbors = |stab_coords: &(usize, usize)| -> Vec<usize> {
            self.get_neighbors(stab_coords)
        };
 
        let run_round = |sim_obj: &mut crate::simulator::StabilizerSimulator, round_p: f64, rng_obj: &mut Xorshift| -> (Vec<bool>, Vec<bool>) {
            let mut round_x = vec![false; num_stabs_x];
            let mut round_z = vec![false; num_stabs_z];
 
            for j in 0..num_stabs_x {
                let q = get_x_stab_qubit(j);
                let m = sim_obj.measure_z(q);
                if m == 1 {
                    sim_obj.apply_x(q);
                }
                inject_single_qubit_noise(sim_obj, q, round_p, bias, rng_obj);
                sim_obj.apply_h(q);
                inject_single_qubit_noise(sim_obj, q, round_p, bias, rng_obj);
            }
            for k in 0..num_stabs_z {
                let q = get_z_stab_qubit(k);
                let m = sim_obj.measure_z(q);
                if m == 1 {
                    sim_obj.apply_x(q);
                }
                inject_single_qubit_noise(sim_obj, q, round_p, bias, rng_obj);
            }
 
            for step in 0..4 {
                for j in 0..num_stabs_x {
                    let stab = &self.x_stabilizers[j];
                    let neighbors = get_neighbors(stab);
                    if step < neighbors.len() {
                        let data_idx = neighbors[step];
                        let anc_idx = get_x_stab_qubit(j);
                        sim_obj.apply_cnot(anc_idx, data_idx);
                        inject_two_qubit_noise(sim_obj, anc_idx, data_idx, round_p, bias, rng_obj);
                    }
                }
                for k in 0..num_stabs_z {
                    let stab = &self.z_stabilizers[k];
                    let neighbors = get_neighbors(stab);
                    if step < neighbors.len() {
                        let data_idx = neighbors[step];
                        let anc_idx = get_z_stab_qubit(k);
                        sim_obj.apply_cnot(data_idx, anc_idx);
                        inject_two_qubit_noise(sim_obj, data_idx, anc_idx, round_p, bias, rng_obj);
                    }
                }
            }
 
            for j in 0..num_stabs_x {
                let q = get_x_stab_qubit(j);
                sim_obj.apply_h(q);
                inject_single_qubit_noise(sim_obj, q, round_p, bias, rng_obj);
            }
 
            for j in 0..num_stabs_x {
                let q = get_x_stab_qubit(j);
                let mut m = sim_obj.measure_z(q) == 1;
                if round_p > 0.0 && rng_obj.next_f64() < round_p {
                    m ^= true;
                }
                round_x[j] = m;
            }
            for k in 0..num_stabs_z {
                let q = get_z_stab_qubit(k);
                let mut m = sim_obj.measure_z(q) == 1;
                if round_p > 0.0 && rng_obj.next_f64() < round_p {
                    m ^= true;
                }
                round_z[k] = m;
            }
 
            (round_x, round_z)
        };
 
        // 1. Run a noiseless projection round to get the baseline syndrome
        let (baseline_x, baseline_z) = run_round(&mut sim, 0.0, &mut rng);
 
        // 2. Run the noisy rounds
        let mut measured_x = vec![vec![false; num_stabs_x]; num_rounds];
        let mut measured_z = vec![vec![false; num_stabs_z]; num_rounds];
 
        for t in 0..num_rounds {
            let (rx, rz) = run_round(&mut sim, p, &mut rng);
            measured_x[t] = rx;
            measured_z[t] = rz;
        }
 
        // --- DECODING Z-stabilizers (X errors) ---
        let graph_z = self.build_syndrome_graph(num_rounds, true);
        let mut defects_z = vec![false; graph_z.num_nodes];
        for t in 0..num_rounds {
            for s_idx in 0..num_stabs_z {
                let prev_outcome = if t == 0 { baseline_z[s_idx] } else { measured_z[t - 1][s_idx] };
                let diff = measured_z[t][s_idx] ^ prev_outcome;
                defects_z[s_idx + t * num_stabs_z] = diff;
            }
        }
        let correction_z_edges = decode_by_type(&graph_z, &defects_z, decoder_type);
 
        let mut correction_x_data = vec![false; num_data];
        for edge_idx in correction_z_edges {
            if let Some(q_idx) = graph_z.edge_to_qubit[edge_idx] {
                correction_x_data[q_idx] ^= true;
            }
        }
 
        // --- DECODING X-stabilizers (Z errors) ---
        let graph_x = self.build_syndrome_graph(num_rounds, false);
        let mut defects_x = vec![false; graph_x.num_nodes];
        for t in 0..num_rounds {
            for s_idx in 0..num_stabs_x {
                let prev_outcome = if t == 0 { baseline_x[s_idx] } else { measured_x[t - 1][s_idx] };
                let diff = measured_x[t][s_idx] ^ prev_outcome;
                defects_x[s_idx + t * num_stabs_x] = diff;
            }
        }
        let correction_x_edges = decode_by_type(&graph_x, &defects_x, decoder_type);
 
        let mut correction_z_data = vec![false; num_data];
        for edge_idx in correction_x_edges {
            if let Some(q_idx) = graph_x.edge_to_qubit[edge_idx] {
                correction_z_data[q_idx] ^= true;
            }
        }

        for i in 0..num_data {
            if correction_x_data[i] {
                sim.apply_x(i);
            }
            if correction_z_data[i] {
                sim.apply_z(i);
            }
        }

        if init_state == "plus" {
            let mut logical_x = 0;
            for x_idx in 0..self.d {
                let q_idx = x_idx + self.d * 0;
                logical_x ^= sim.measure_x(q_idx);
            }
            logical_x == 1
        } else {
            let mut logical_z = 0;
            for y_idx in 0..self.d {
                let q_idx = 0 + self.d * y_idx;
                logical_z ^= sim.measure_z(q_idx);
            }
            logical_z == 1
        }
    }
}

pub struct XZZXSurfaceCode {
    pub d: usize,
    pub data_qubits: Vec<(usize, usize)>,
    pub stabilizers: Vec<(usize, usize)>,
    pub z_stabilizers: Vec<(usize, usize)>,
    pub x_stabilizers: Vec<(usize, usize)>,
}

impl XZZXSurfaceCode {
    pub fn new(d: usize) -> Self {
        let mut data_qubits = Vec::new();
        for y in (1..(2 * d)).step_by(2) {
            for x in (1..(2 * d)).step_by(2) {
                data_qubits.push((x, y));
            }
        }

        let mut stabilizers = Vec::new();
        let mut z_stabilizers = Vec::new();
        let mut x_stabilizers = Vec::new();
        // Construct Z-like coordinates of standard rotated surface code
        for y in (0..=(2 * d)).step_by(2) {
            for x in (2..(2 * d)).step_by(2) {
                if ((x + y) / 2) % 2 == 0 {
                    z_stabilizers.push((x, y));
                    stabilizers.push((x, y));
                }
            }
        }
        // Construct X-like coordinates of standard rotated surface code
        for y in (2..(2 * d)).step_by(2) {
            for x in (0..=(2 * d)).step_by(2) {
                if ((x + y) / 2) % 2 == 1 {
                    x_stabilizers.push((x, y));
                    stabilizers.push((x, y));
                }
            }
        }

        XZZXSurfaceCode {
            d,
            data_qubits,
            stabilizers,
            z_stabilizers,
            x_stabilizers,
        }
    }

    pub fn get_neighbor_idx(&self, x: i32, y: i32) -> Option<usize> {
        if x >= 1 && x < (2 * self.d) as i32 && y >= 1 && y < (2 * self.d) as i32 {
            let x_idx = (x - 1) as usize / 2;
            let y_idx = (y - 1) as usize / 2;
            Some(x_idx + self.d * y_idx)
        } else {
            None
        }
    }

    pub fn get_neighbors(&self, stab: &(usize, usize)) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let (sx, sy) = *stab;
        for &(dx, dy) in &[
            (sx as i32 - 1, sy as i32 - 1),
            (sx as i32 - 1, sy as i32 + 1),
            (sx as i32 + 1, sy as i32 - 1),
            (sx as i32 + 1, sy as i32 + 1),
        ] {
            if let Some(idx) = self.get_neighbor_idx(dx, dy) {
                neighbors.push(idx);
            }
        }
        neighbors
    }

    pub fn build_syndrome_graph(&self, num_rounds: usize, is_for_x_errors: bool) -> SyndromeGraph {
        let num_stabs = self.stabilizers.len();
        let num_nodes = num_stabs * num_rounds;

        let mut edges = Vec::new();
        let mut edge_to_qubit = Vec::new();
        let mut edge_id = 0;

        for t in 0..num_rounds {
            for q_idx in 0..self.data_qubits.len() {
                let (qx, qy) = self.data_qubits[q_idx];
                let mut connected_stabs = Vec::new();
                for (s_idx, stab) in self.stabilizers.iter().enumerate() {
                    let (sx, sy) = *stab;
                    let is_connected = if is_for_x_errors {
                        (qx as i32 == sx as i32 + 1 && qy as i32 == sy as i32 - 1) ||
                        (qx as i32 == sx as i32 - 1 && qy as i32 == sy as i32 + 1)
                    } else {
                        (qx as i32 == sx as i32 - 1 && qy as i32 == sy as i32 - 1) ||
                        (qx as i32 == sx as i32 + 1 && qy as i32 == sy as i32 + 1)
                    };
                    if is_connected {
                        connected_stabs.push(s_idx);
                    }
                }

                if connected_stabs.len() == 2 {
                    let u = connected_stabs[0] + t * num_stabs;
                    let v = connected_stabs[1] + t * num_stabs;
                    edges.push(Edge { u, v, id: edge_id });
                    edge_to_qubit.push(Some(q_idx));
                    edge_id += 1;
                } else if connected_stabs.len() == 1 {
                    let u = connected_stabs[0] + t * num_stabs;
                    let v = num_nodes;
                    edges.push(Edge { u, v, id: edge_id });
                    edge_to_qubit.push(Some(q_idx));
                    edge_id += 1;
                }
            }
        }

        for t in 0..(num_rounds - 1) {
            for s_idx in 0..num_stabs {
                let u = s_idx + t * num_stabs;
                let v = s_idx + (t + 1) * num_stabs;
                edges.push(Edge { u, v, id: edge_id });
                edge_to_qubit.push(None);
                edge_id += 1;
            }
        }

        SyndromeGraph {
            num_nodes,
            edges,
            edge_to_qubit,
        }
    }

    pub fn simulate_phenomenological_noise(&self, num_rounds: usize, p: f64, bias: f64, decoder_type: usize) -> bool {
        #[cfg(feature = "python")]
        let mut rng = Xorshift::new(rand::random());
        #[cfg(not(feature = "python"))]
        let mut rng = Xorshift::new(12345);

        let num_stabs = self.stabilizers.len();
        let num_data = self.data_qubits.len();

        let mut physical_x = vec![false; num_data];
        let mut physical_z = vec![false; num_data];

        let mut measured = vec![vec![false; num_stabs]; num_rounds];

        for t in 0..num_rounds {
            for q in 0..num_data {
                let (err_x, err_z) = sample_biased_error(p, bias, &mut rng);
                if err_x { physical_x[q] ^= true; }
                if err_z { physical_z[q] ^= true; }
            }

            for s_idx in 0..num_stabs {
                let (sx, sy) = self.stabilizers[s_idx];
                let mut parity = false;
                
                if let Some(q) = self.get_neighbor_idx(sx as i32 - 1, sy as i32 - 1) {
                    if physical_z[q] { parity ^= true; }
                }
                if let Some(q) = self.get_neighbor_idx(sx as i32 + 1, sy as i32 - 1) {
                    if physical_x[q] { parity ^= true; }
                }
                if let Some(q) = self.get_neighbor_idx(sx as i32 - 1, sy as i32 + 1) {
                    if physical_x[q] { parity ^= true; }
                }
                if let Some(q) = self.get_neighbor_idx(sx as i32 + 1, sy as i32 + 1) {
                    if physical_z[q] { parity ^= true; }
                }

                if rng.next_f64() < p {
                    parity ^= true;
                }
                measured[t][s_idx] = parity;
            }
        }

        let graph_z = self.build_syndrome_graph(num_rounds, true);
        let mut defects_z = vec![false; graph_z.num_nodes];
        for t in 0..num_rounds {
            for s_idx in 0..num_stabs {
                let prev_outcome = if t == 0 { false } else { measured[t - 1][s_idx] };
                let diff = measured[t][s_idx] ^ prev_outcome;
                defects_z[s_idx + t * num_stabs] = diff;
            }
        }
        let correction_z_edges = decode_by_type(&graph_z, &defects_z, decoder_type);

        let mut correction_x_data = vec![false; num_data];
        for edge_idx in correction_z_edges {
            if let Some(q_idx) = graph_z.edge_to_qubit[edge_idx] {
                correction_x_data[q_idx] ^= true;
            }
        }

        let graph_x = self.build_syndrome_graph(num_rounds, false);
        let defects_x = defects_z.clone();
        let correction_x_edges = decode_by_type(&graph_x, &defects_x, decoder_type);

        let mut correction_z_data = vec![false; num_data];
        for edge_idx in correction_x_edges {
            if let Some(q_idx) = graph_x.edge_to_qubit[edge_idx] {
                correction_z_data[q_idx] ^= true;
            }
        }

        let mut residual_x = vec![false; num_data];
        let mut residual_z = vec![false; num_data];
        for q in 0..num_data {
            residual_x[q] = physical_x[q] ^ correction_x_data[q];
            residual_z[q] = physical_z[q] ^ correction_z_data[q];
        }

        let mut logical_err_1 = false;
        for x_idx in 0..self.d {
            let q = x_idx + self.d * 0;
            let err = if x_idx % 2 == 0 { residual_z[q] } else { residual_x[q] };
            if err {
                logical_err_1 ^= true;
            }
        }

        let mut logical_err_2 = false;
        for y_idx in 0..self.d {
            let q = 0 + self.d * y_idx;
            let err = if y_idx % 2 == 0 { residual_x[q] } else { residual_z[q] };
            if err {
                logical_err_2 ^= true;
            }
        }

        logical_err_1 || logical_err_2
    }

    pub fn simulate_data_noise(&self, p: f64, bias: f64, decoder_type: usize) -> bool {
        #[cfg(feature = "python")]
        let mut rng = Xorshift::new(rand::random());
        #[cfg(not(feature = "python"))]
        let mut rng = Xorshift::new(12345);

        let num_data = self.data_qubits.len();
        let mut physical_x = vec![false; num_data];
        let mut physical_z = vec![false; num_data];
        for q in 0..num_data {
            let (err_x, err_z) = sample_biased_error(p, bias, &mut rng);
            if err_x { physical_x[q] = true; }
            if err_z { physical_z[q] = true; }
        }

        let num_stabs = self.stabilizers.len();
        let mut measured = vec![false; num_stabs];
        for s_idx in 0..num_stabs {
            let (sx, sy) = self.stabilizers[s_idx];
            let mut parity = false;
            if let Some(q) = self.get_neighbor_idx(sx as i32 - 1, sy as i32 - 1) {
                if physical_z[q] { parity ^= true; }
            }
            if let Some(q) = self.get_neighbor_idx(sx as i32 + 1, sy as i32 - 1) {
                if physical_x[q] { parity ^= true; }
            }
            if let Some(q) = self.get_neighbor_idx(sx as i32 - 1, sy as i32 + 1) {
                if physical_x[q] { parity ^= true; }
            }
            if let Some(q) = self.get_neighbor_idx(sx as i32 + 1, sy as i32 + 1) {
                if physical_z[q] { parity ^= true; }
            }
            measured[s_idx] = parity;
        }

        let graph_z = self.build_syndrome_graph(1, true);
        let correction_z_edges = decode_by_type(&graph_z, &measured, decoder_type);
        let mut correction_x_data = vec![false; num_data];
        for edge_idx in correction_z_edges {
            if let Some(q_idx) = graph_z.edge_to_qubit[edge_idx] {
                correction_x_data[q_idx] ^= true;
            }
        }

        let graph_x = self.build_syndrome_graph(1, false);
        let correction_x_edges = decode_by_type(&graph_x, &measured, decoder_type);
        let mut correction_z_data = vec![false; num_data];
        for edge_idx in correction_x_edges {
            if let Some(q_idx) = graph_x.edge_to_qubit[edge_idx] {
                correction_z_data[q_idx] ^= true;
            }
        }

        let mut residual_x = vec![false; num_data];
        let mut residual_z = vec![false; num_data];
        for q in 0..num_data {
            residual_x[q] = physical_x[q] ^ correction_x_data[q];
            residual_z[q] = physical_z[q] ^ correction_z_data[q];
        }

        let mut logical_err_1 = false;
        for x_idx in 0..self.d {
            let q = x_idx + self.d * 0;
            let err = if x_idx % 2 == 0 { residual_z[q] } else { residual_x[q] };
            if err {
                logical_err_1 ^= true;
            }
        }

        let mut logical_err_2 = false;
        for y_idx in 0..self.d {
            let q = 0 + self.d * y_idx;
            let err = if y_idx % 2 == 0 { residual_x[q] } else { residual_z[q] };
            if err {
                logical_err_2 ^= true;
            }
        }

        logical_err_1 || logical_err_2
    }

    pub fn simulate_circuit_noise(&self, num_rounds: usize, p: f64, bias: f64, init_state: &str, decoder_type: usize) -> bool {
        let num_data = self.data_qubits.len();
        let num_stabs_z = self.z_stabilizers.len();
        let num_stabs_x = self.x_stabilizers.len();
        let total_qubits = num_data + num_stabs_z + num_stabs_x;
        
        let mut sim = crate::simulator::StabilizerSimulator::new(total_qubits);
        #[cfg(feature = "python")]
        let mut rng = Xorshift::new(rand::random());
        #[cfg(not(feature = "python"))]
        let mut rng = Xorshift::new(54321);

        let get_stab_qubit = |j: usize| num_data + j;

        if init_state == "plus" {
            for x_idx in 0..self.d {
                let q_idx = x_idx + self.d * 0;
                sim.apply_h(q_idx);
            }
            for i in 0..num_data {
                inject_single_qubit_noise(&mut sim, i, p, bias, &mut rng);
            }
        } else if init_state == "y" {
            sim.apply_h(0);
            sim.apply_s(0);
            for x_idx in 1..self.d {
                sim.apply_h(x_idx);
            }
            for i in 0..num_data {
                inject_single_qubit_noise(&mut sim, i, p, bias, &mut rng);
            }
        } else {
            for i in 0..num_data {
                inject_single_qubit_noise(&mut sim, i, p, bias, &mut rng);
            }
        }

        // Helper to run one round of syndrome extraction
        let run_round = |sim_obj: &mut crate::simulator::StabilizerSimulator, round_p: f64, rng_obj: &mut Xorshift| -> (Vec<bool>, Vec<bool>) {
            let mut round_z = vec![false; num_stabs_z];
            let mut round_x = vec![false; num_stabs_x];

            // 1. Z stabilizers (detect X errors, measure Z-stabilizers, connect via CNOT control=data, target=ancilla)
            for j in 0..num_stabs_z {
                let q = get_stab_qubit(j);
                let m = sim_obj.measure_z(q);
                if m == 1 {
                    sim_obj.apply_x(q);
                }
                inject_single_qubit_noise(sim_obj, q, round_p, bias, rng_obj);
            }

            // apply CNOT gates for Z stabilizers
            for step in 0..4 {
                for j in 0..num_stabs_z {
                    let (sx, sy) = self.z_stabilizers[j];
                    let dx_dy = match step {
                        0 => (sx as i32 - 1, sy as i32 - 1),
                        1 => (sx as i32 + 1, sy as i32 - 1),
                        2 => (sx as i32 - 1, sy as i32 + 1),
                        3 => (sx as i32 + 1, sy as i32 + 1),
                        _ => unreachable!(),
                    };
                    if let Some(data_idx) = self.get_neighbor_idx(dx_dy.0, dx_dy.1) {
                        let anc_idx = get_stab_qubit(j);
                        sim_obj.apply_cnot(data_idx, anc_idx);
                        inject_two_qubit_noise(sim_obj, data_idx, anc_idx, round_p, bias, rng_obj);
                    }
                }
            }

            for j in 0..num_stabs_z {
                let q = get_stab_qubit(j);
                let mut m = sim_obj.measure_z(q) == 1;
                if round_p > 0.0 && rng_obj.next_f64() < round_p {
                    m ^= true;
                }
                round_z[j] = m;
            }

            // 2. X stabilizers (detect Z errors, measure X-stabilizers, connect via CNOT control=ancilla, target=data)
            for j in 0..num_stabs_x {
                let q = get_stab_qubit(num_stabs_z + j);
                let m = sim_obj.measure_z(q);
                if m == 1 {
                    sim_obj.apply_x(q);
                }
                sim_obj.apply_h(q);
                inject_single_qubit_noise(sim_obj, q, round_p, bias, rng_obj);
            }

            for step in 0..4 {
                for j in 0..num_stabs_x {
                    let (sx, sy) = self.x_stabilizers[j];
                    let dx_dy = match step {
                        0 => (sx as i32 - 1, sy as i32 - 1),
                        1 => (sx as i32 + 1, sy as i32 - 1),
                        2 => (sx as i32 - 1, sy as i32 + 1),
                        3 => (sx as i32 + 1, sy as i32 + 1),
                        _ => unreachable!(),
                    };
                    if let Some(data_idx) = self.get_neighbor_idx(dx_dy.0, dx_dy.1) {
                        let anc_idx = get_stab_qubit(num_stabs_z + j);
                        sim_obj.apply_cnot(anc_idx, data_idx);
                        inject_two_qubit_noise(sim_obj, anc_idx, data_idx, round_p, bias, rng_obj);
                    }
                }
            }

            for j in 0..num_stabs_x {
                let q = get_stab_qubit(num_stabs_z + j);
                sim_obj.apply_h(q);
                let mut m = sim_obj.measure_z(q) == 1;
                if round_p > 0.0 && rng_obj.next_f64() < round_p {
                    m ^= true;
                }
                round_x[j] = m;
            }

            (round_z, round_x)
        };

        // 1. Run noiseless projection round to get baseline syndrome
        let (baseline_z, baseline_x) = run_round(&mut sim, 0.0, &mut rng);

        // 2. Run noisy rounds
        let mut measured_z = vec![vec![false; num_stabs_z]; num_rounds];
        let mut measured_x = vec![vec![false; num_stabs_x]; num_rounds];
        for t in 0..num_rounds {
            let (rz, rx) = run_round(&mut sim, p, &mut rng);
            measured_z[t] = rz;
            measured_x[t] = rx;
        }

        let graph_z = self.build_syndrome_graph(num_rounds, true);
        let mut defects_z = vec![false; graph_z.num_nodes];
        for t in 0..num_rounds {
            for s_idx in 0..num_stabs_z {
                let prev_outcome = if t == 0 { baseline_z[s_idx] } else { measured_z[t - 1][s_idx] };
                let diff = measured_z[t][s_idx] ^ prev_outcome;
                defects_z[s_idx + t * num_stabs_z] = diff;
            }
        }
        let correction_z_edges = decode_by_type(&graph_z, &defects_z, decoder_type);

        let mut correction_x_data = vec![false; num_data];
        for edge_idx in correction_z_edges {
            if let Some(q_idx) = graph_z.edge_to_qubit[edge_idx] {
                correction_x_data[q_idx] ^= true;
            }
        }

        let graph_x = self.build_syndrome_graph(num_rounds, false);
        let mut defects_x = vec![false; graph_x.num_nodes];
        for t in 0..num_rounds {
            for s_idx in 0..num_stabs_x {
                let prev_outcome = if t == 0 { baseline_x[s_idx] } else { measured_x[t - 1][s_idx] };
                let diff = measured_x[t][s_idx] ^ prev_outcome;
                defects_x[s_idx + t * num_stabs_x] = diff;
            }
        }
        let correction_x_edges = decode_by_type(&graph_x, &defects_x, decoder_type);

        let mut correction_z_data = vec![false; num_data];
        for edge_idx in correction_x_edges {
            if let Some(q_idx) = graph_x.edge_to_qubit[edge_idx] {
                correction_z_data[q_idx] ^= true;
            }
        }

        for i in 0..num_data {
            if correction_x_data[i] {
                sim.apply_x(i);
            }
            if correction_z_data[i] {
                sim.apply_z(i);
            }
        }

        if init_state == "plus" {
            let mut logical_err = false;
            for x_idx in 0..self.d {
                let q_idx = x_idx + self.d * 0;
                let outcome = sim.measure_x(q_idx);
                if outcome == 1 {
                    logical_err ^= true;
                }
            }
            logical_err
        } else if init_state == "y" {
            let mut logical_err = sim.measure_y(0) == 1;
            for x_idx in 1..self.d {
                if sim.measure_x(x_idx) == 1 {
                    logical_err ^= true;
                }
            }
            for y_idx in 1..self.d {
                if sim.measure_z(self.d * y_idx) == 1 {
                    logical_err ^= true;
                }
            }
            logical_err
        } else {
            let mut logical_err = false;
            for y_idx in 0..self.d {
                let q_idx = 0 + self.d * y_idx;
                let outcome = sim.measure_z(q_idx);
                if outcome == 1 {
                    logical_err ^= true;
                }
            }
            logical_err
        }
    }
}

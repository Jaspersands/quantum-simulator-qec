use crate::decoder::{Edge, SyndromeGraph, decode_union_find, decode_greedy};

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

fn inject_single_qubit_noise(sim: &mut crate::simulator::StabilizerSimulator, qubit: usize, p: f64, rng: &mut Xorshift) {
    if rng.next_f64() < p {
        let choice = rng.next_u64() % 3;
        match choice {
            0 => sim.apply_x(qubit),
            1 => sim.apply_y(qubit),
            2 => sim.apply_z(qubit),
            _ => unreachable!(),
        }
    }
}

fn inject_two_qubit_noise(sim: &mut crate::simulator::StabilizerSimulator, q1: usize, q2: usize, p: f64, rng: &mut Xorshift) {
    if rng.next_f64() < p {
        let choice = rng.next_u64() % 15;
        let val = choice + 1;
        let e1 = val / 4;
        let e2 = val % 4;
        match e1 {
            1 => sim.apply_x(q1),
            2 => sim.apply_y(q1),
            3 => sim.apply_z(q1),
            _ => {}
        }
        match e2 {
            1 => sim.apply_x(q2),
            2 => sim.apply_y(q2),
            3 => sim.apply_z(q2),
            _ => {}
        }
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
    pub fn simulate_phenomenological_noise(&self, num_rounds: usize, p: f64) -> bool {
        #[cfg(feature = "python")]
        let mut rng = Xorshift::new(rand::random());
        #[cfg(not(feature = "python"))]
        let mut rng = Xorshift::new(12345);

        let num_stabs_z = self.z_stabilizers.len();
        let num_stabs_x = self.x_stabilizers.len();
        let num_data = self.data_qubits.len();

        // Accumulator for physical errors on data qubits (X and Z)
        let mut physical_x = vec![false; num_data];
        let mut physical_z = vec![false; num_data];

        // Store syndromes for all rounds
        // Z-type stabilizers detect X errors
        let mut measured_z = vec![vec![false; num_stabs_z]; num_rounds];
        // X-type stabilizers detect Z errors
        let mut measured_x = vec![vec![false; num_stabs_x]; num_rounds];

        // Run the simulation round by round
        for t in 0..num_rounds {
            // 1. Inject physical data qubit errors (X and Z)
            for q in 0..num_data {
                if rng.next_f64() < p {
                    physical_x[q] ^= true;
                }
                if rng.next_f64() < p {
                    physical_z[q] ^= true;
                }
            }

            // 2. Measure stabilizers
            // Z-stabilizers
            for s_idx in 0..num_stabs_z {
                let neighbors = self.get_neighbors(&self.z_stabilizers[s_idx]);
                let mut parity = false;
                for &q in &neighbors {
                    if physical_x[q] {
                        parity ^= true;
                    }
                }
                // Measurement error
                if rng.next_f64() < p {
                    parity ^= true;
                }
                measured_z[t][s_idx] = parity;
            }

            // X-stabilizers
            for s_idx in 0..num_stabs_x {
                let neighbors = self.get_neighbors(&self.x_stabilizers[s_idx]);
                let mut parity = false;
                for &q in &neighbors {
                    if physical_z[q] {
                        parity ^= true;
                    }
                }
                // Measurement error
                if rng.next_f64() < p {
                    parity ^= true;
                }
                measured_x[t][s_idx] = parity;
            }
        }

        // --- DECODING Z-stabilizers (X errors) ---
        let graph_z = self.build_syndrome_graph(num_rounds, true);
        let mut defects_z = vec![false; graph_z.num_nodes];
        for t in 0..num_rounds {
            for s_idx in 0..num_stabs_z {
                let prev_outcome = if t == 0 { false } else { measured_z[t - 1][s_idx] };
                let diff = measured_z[t][s_idx] ^ prev_outcome;
                defects_z[s_idx + t * num_stabs_z] = diff;
            }
        }
        let correction_z_edges = decode_union_find(&graph_z, &defects_z);

        // Apply Z corrections
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
                let prev_outcome = if t == 0 { false } else { measured_x[t - 1][s_idx] };
                let diff = measured_x[t][s_idx] ^ prev_outcome;
                defects_x[s_idx + t * num_stabs_x] = diff;
            }
        }
        let correction_x_edges = decode_union_find(&graph_x, &defects_x);

        // Apply X corrections
        let mut correction_z_data = vec![false; num_data];
        for edge_idx in correction_x_edges {
            if let Some(q_idx) = graph_x.edge_to_qubit[edge_idx] {
                correction_z_data[q_idx] ^= true;
            }
        }

        // --- LOGICAL ERROR VERIFICATION ---
        // Residual errors on data qubits
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
    pub fn simulate_data_noise(&self, p: f64) -> bool {
        #[cfg(feature = "python")]
        let mut rng = Xorshift::new(rand::random());
        #[cfg(not(feature = "python"))]
        let mut rng = Xorshift::new(12345);

        let num_data = self.data_qubits.len();
        let mut physical_x = vec![false; num_data];
        let mut physical_z = vec![false; num_data];
        
        for q in 0..num_data {
            if rng.next_f64() < p {
                physical_x[q] = true;
            }
            if rng.next_f64() < p {
                physical_z[q] = true;
            }
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
        let correction_z_edges = decode_union_find(&graph_z, &measured_z);
        let mut correction_x_data = vec![false; num_data];
        for edge_idx in correction_z_edges {
            if let Some(q_idx) = graph_z.edge_to_qubit[edge_idx] {
                correction_x_data[q_idx] ^= true;
            }
        }

        let graph_x = self.build_syndrome_graph(1, false);
        let correction_x_edges = decode_union_find(&graph_x, &measured_x);
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

    pub fn simulate_circuit_noise(&self, num_rounds: usize, p: f64, init_state: &str, use_greedy: bool) -> bool {
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
                inject_single_qubit_noise(&mut sim, i, p, &mut rng);
            }
        } else {
            for i in 0..num_data {
                inject_single_qubit_noise(&mut sim, i, p, &mut rng);
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
                inject_single_qubit_noise(sim_obj, q, round_p, rng_obj);
                sim_obj.apply_h(q);
                inject_single_qubit_noise(sim_obj, q, round_p, rng_obj);
            }
            for k in 0..num_stabs_z {
                let q = get_z_stab_qubit(k);
                let m = sim_obj.measure_z(q);
                if m == 1 {
                    sim_obj.apply_x(q);
                }
                inject_single_qubit_noise(sim_obj, q, round_p, rng_obj);
            }

            for step in 0..4 {
                for j in 0..num_stabs_x {
                    let stab = &self.x_stabilizers[j];
                    let neighbors = get_neighbors(stab);
                    if step < neighbors.len() {
                        let data_idx = neighbors[step];
                        let anc_idx = get_x_stab_qubit(j);
                        sim_obj.apply_cnot(anc_idx, data_idx);
                        inject_two_qubit_noise(sim_obj, anc_idx, data_idx, round_p, rng_obj);
                    }
                }
                for k in 0..num_stabs_z {
                    let stab = &self.z_stabilizers[k];
                    let neighbors = get_neighbors(stab);
                    if step < neighbors.len() {
                        let data_idx = neighbors[step];
                        let anc_idx = get_z_stab_qubit(k);
                        sim_obj.apply_cnot(data_idx, anc_idx);
                        inject_two_qubit_noise(sim_obj, data_idx, anc_idx, round_p, rng_obj);
                    }
                }
            }

            for j in 0..num_stabs_x {
                let q = get_x_stab_qubit(j);
                sim_obj.apply_h(q);
                inject_single_qubit_noise(sim_obj, q, round_p, rng_obj);
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
        // Since we want this to be noiseless, we clone the simulator first,
        // project it noiselessly, and then discard the clone!
        // Wait, no! If we project it on a clone, the projection collapsed the state
        // on the clone, but the main simulator is still unprojected!
        // To get the baseline syndrome, we project the MAIN simulator noiselessly (round p = 0.0).
        // This is perfectly fine, because a noiseless projection collapses the state into the code space without introducing errors!
        // So the main simulator is now in a clean projected code state.
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
        let correction_z_edges = if use_greedy {
            decode_greedy(&graph_z, &defects_z)
        } else {
            decode_union_find(&graph_z, &defects_z)
        };

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
        let correction_x_edges = if use_greedy {
            decode_greedy(&graph_x, &defects_x)
        } else {
            decode_union_find(&graph_x, &defects_x)
        };

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
        // Construct Z-like coordinates of standard rotated surface code
        for y in (0..=(2 * d)).step_by(2) {
            for x in (2..(2 * d)).step_by(2) {
                if ((x + y) / 2) % 2 == 0 {
                    stabilizers.push((x, y));
                }
            }
        }
        // Construct X-like coordinates of standard rotated surface code
        for y in (2..(2 * d)).step_by(2) {
            for x in (0..=(2 * d)).step_by(2) {
                if ((x + y) / 2) % 2 == 1 {
                    stabilizers.push((x, y));
                }
            }
        }

        XZZXSurfaceCode {
            d,
            data_qubits,
            stabilizers,
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

    pub fn simulate_phenomenological_noise(&self, num_rounds: usize, p: f64) -> bool {
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
                if rng.next_f64() < p {
                    physical_x[q] ^= true;
                }
                if rng.next_f64() < p {
                    physical_z[q] ^= true;
                }
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
        let correction_z_edges = decode_union_find(&graph_z, &defects_z);

        let mut correction_x_data = vec![false; num_data];
        for edge_idx in correction_z_edges {
            if let Some(q_idx) = graph_z.edge_to_qubit[edge_idx] {
                correction_x_data[q_idx] ^= true;
            }
        }

        let graph_x = self.build_syndrome_graph(num_rounds, false);
        let defects_x = defects_z.clone();
        let correction_x_edges = decode_union_find(&graph_x, &defects_x);

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

    pub fn simulate_data_noise(&self, p: f64) -> bool {
        #[cfg(feature = "python")]
        let mut rng = Xorshift::new(rand::random());
        #[cfg(not(feature = "python"))]
        let mut rng = Xorshift::new(12345);

        let num_data = self.data_qubits.len();
        let mut physical_x = vec![false; num_data];
        let mut physical_z = vec![false; num_data];
        for q in 0..num_data {
            if rng.next_f64() < p {
                physical_x[q] = true;
            }
            if rng.next_f64() < p {
                physical_z[q] = true;
            }
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
        let correction_z_edges = decode_union_find(&graph_z, &measured);
        let mut correction_x_data = vec![false; num_data];
        for edge_idx in correction_z_edges {
            if let Some(q_idx) = graph_z.edge_to_qubit[edge_idx] {
                correction_x_data[q_idx] ^= true;
            }
        }

        let graph_x = self.build_syndrome_graph(1, false);
        let correction_x_edges = decode_union_find(&graph_x, &measured);
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

    pub fn simulate_circuit_noise(&self, num_rounds: usize, p: f64, init_state: &str, use_greedy: bool) -> bool {
        let num_data = self.data_qubits.len();
        let num_stabs = self.stabilizers.len();
        let total_qubits = num_data + num_stabs;
        
        let mut sim = crate::simulator::StabilizerSimulator::new(total_qubits);
        #[cfg(feature = "python")]
        let mut rng = Xorshift::new(rand::random());
        #[cfg(not(feature = "python"))]
        let mut rng = Xorshift::new(54321);

        let get_stab_qubit = |j: usize| num_data + j;

        if init_state == "plus" {
            for x_idx in 0..self.d {
                let q_idx = x_idx + self.d * 0;
                if x_idx % 2 == 0 {
                    sim.apply_h(q_idx);
                }
            }
            for i in 0..num_data {
                inject_single_qubit_noise(&mut sim, i, p, &mut rng);
            }
        } else {
            for y_idx in 0..self.d {
                let q_idx = 0 + self.d * y_idx;
                if y_idx % 2 == 1 {
                    sim.apply_h(q_idx);
                }
            }
            for i in 0..num_data {
                inject_single_qubit_noise(&mut sim, i, p, &mut rng);
            }
        }

        // Helper to run one round of syndrome extraction
        let run_round = |sim_obj: &mut crate::simulator::StabilizerSimulator, round_p: f64, rng_obj: &mut Xorshift| -> Vec<bool> {
            let mut round_out = vec![false; num_stabs];

            for j in 0..num_stabs {
                let q = get_stab_qubit(j);
                let m = sim_obj.measure_z(q);
                if m == 1 {
                    sim_obj.apply_x(q);
                }
                inject_single_qubit_noise(sim_obj, q, round_p, rng_obj);
            }

            for step in 0..4 {
                for j in 0..num_stabs {
                    let (sx, sy) = self.stabilizers[j];
                    let dx_dy = match step {
                        0 => (sx as i32 - 1, sy as i32 - 1, true),
                        1 => (sx as i32 + 1, sy as i32 - 1, false),
                        2 => (sx as i32 - 1, sy as i32 + 1, false),
                        3 => (sx as i32 + 1, sy as i32 + 1, true),
                        _ => unreachable!(),
                    };

                    if let Some(data_idx) = self.get_neighbor_idx(dx_dy.0, dx_dy.1) {
                        let anc_idx = get_stab_qubit(j);
                        if dx_dy.2 {
                            sim_obj.apply_h(data_idx);
                            inject_single_qubit_noise(sim_obj, data_idx, round_p, rng_obj);
                        }
                        sim_obj.apply_cnot(data_idx, anc_idx);
                        inject_two_qubit_noise(sim_obj, data_idx, anc_idx, round_p, rng_obj);
                        if dx_dy.2 {
                            sim_obj.apply_h(data_idx);
                            inject_single_qubit_noise(sim_obj, data_idx, round_p, rng_obj);
                        }
                    }
                }
            }

            for j in 0..num_stabs {
                let q = get_stab_qubit(j);
                let mut m = sim_obj.measure_z(q) == 1;
                if round_p > 0.0 && rng_obj.next_f64() < round_p {
                    m ^= true;
                }
                round_out[j] = m;
            }

            round_out
        };

        // 1. Run noiseless projection round to get baseline syndrome
        let baseline = run_round(&mut sim, 0.0, &mut rng);

        // 2. Run noisy rounds
        let mut measured = vec![vec![false; num_stabs]; num_rounds];
        for t in 0..num_rounds {
            measured[t] = run_round(&mut sim, p, &mut rng);
        }

        let graph_z = self.build_syndrome_graph(num_rounds, true);
        let mut defects_z = vec![false; graph_z.num_nodes];
        for t in 0..num_rounds {
            for s_idx in 0..num_stabs {
                let prev_outcome = if t == 0 { baseline[s_idx] } else { measured[t - 1][s_idx] };
                let diff = measured[t][s_idx] ^ prev_outcome;
                defects_z[s_idx + t * num_stabs] = diff;
            }
        }
        let correction_z_edges = if use_greedy {
            decode_greedy(&graph_z, &defects_z)
        } else {
            decode_union_find(&graph_z, &defects_z)
        };

        let mut correction_x_data = vec![false; num_data];
        for edge_idx in correction_z_edges {
            if let Some(q_idx) = graph_z.edge_to_qubit[edge_idx] {
                correction_x_data[q_idx] ^= true;
            }
        }

        let graph_x = self.build_syndrome_graph(num_rounds, false);
        let defects_x = defects_z.clone();
        let correction_x_edges = if use_greedy {
            decode_greedy(&graph_x, &defects_x)
        } else {
            decode_union_find(&graph_x, &defects_x)
        };

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
            let mut logical_err = 0;
            for x_idx in 0..self.d {
                let q_idx = x_idx + self.d * 0;
                let outcome = if x_idx % 2 == 0 {
                    sim.measure_x(q_idx)
                } else {
                    sim.measure_z(q_idx)
                };
                logical_err ^= outcome;
            }
            logical_err == 1
        } else {
            let mut logical_err = 0;
            for y_idx in 0..self.d {
                let q_idx = 0 + self.d * y_idx;
                let outcome = if y_idx % 2 == 0 {
                    sim.measure_z(q_idx)
                } else {
                    sim.measure_x(q_idx)
                };
                logical_err ^= outcome;
            }
            logical_err == 1
        }
    }
}

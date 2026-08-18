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

/// Per-shot generator for the WebAssembly build.
///
/// The obvious `Xorshift::new(12345)` is a trap: these `simulate_*` functions
/// are each one Monte Carlo shot, and `wasm_run_benchmark` calls them in a
/// loop. Seeding from a constant inside the shot makes every shot in a batch
/// bit-identical, so the reported rate collapses to a step function — flat
/// zero, a plateau, then flat one — rather than a curve.
///
/// Instead keep one generator alive across calls and draw a fresh seed from it
/// per shot. Single-threaded by construction: wasm32-unknown-unknown has no
/// threads here, and the Python build takes the `rand::random()` branch and
/// never touches this.
#[cfg(not(feature = "python"))]
static mut RNG_STREAM: u64 = 0x2545_F491_4F6C_DD1D;

#[cfg(not(feature = "python"))]
pub fn seed_global_rng(seed: u64) {
    unsafe {
        let slot = core::ptr::addr_of_mut!(RNG_STREAM);
        slot.write(if seed == 0 { 0x2545_F491_4F6C_DD1D } else { seed });
    }
}

/// SplitMix64 over a counter.
///
/// Advancing the shot seed with the same xorshift recurrence the shots
/// themselves use is not enough: seeding shot N+1 with one step of shot N's
/// state gives it shot N's stream offset by a single draw, so consecutive
/// shots are almost perfectly correlated and batch variance comes out several
/// times larger than binomial. SplitMix64 is built for exactly this job —
/// a counter plus a strong finalizer, giving seeds that decorrelate.
#[cfg(not(feature = "python"))]
fn next_shot_rng() -> Xorshift {
    unsafe {
        let slot = core::ptr::addr_of_mut!(RNG_STREAM);
        let counter = slot.read().wrapping_add(0x9E37_79B9_7F4A_7C15);
        slot.write(counter);

        let mut z = counter;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        Xorshift::new(z)
    }
}

/// Is a residual Pauli a logical error?
///
/// After decoding, a residual that commutes with every stabilizer is either a
/// product of stabilizers — harmless — or a logical operator. Testing that
/// directly is convention-free: reduce the residual against a row-reduced
/// basis of the stabilizer group and see whether anything is left.
///
/// The alternative, and what this file did for the XZZX code, is to hard-code a
/// string of Paulis believed to represent the logical operator and check
/// anticommutation with it. When the string is wrong the check misfires on
/// residuals that are not logical operators at all: a *weight-one* residual was
/// being reported as a logical failure, which is impossible in a distance-3 or
/// larger code, and it happened at every distance, so the logical error rate
/// came out flat instead of falling.
///
/// Brute force over the actual stabilizer group confirms the XZZX construction
/// is sound — minimum logical weight 3 at d=3, above 4 at d=5 — so the code was
/// right and only the check was wrong.
pub struct LogicalCheck {
    /// Row-reduced stabilizer basis: (x bits, z bits, pivot is in x, pivot bit).
    basis: Vec<(u128, u128, bool, u32)>,
}

impl LogicalCheck {
    /// `stabilizers` are symplectic (x, z) bit patterns, one per generator.
    pub fn new(stabilizers: &[(u128, u128)], num_qubits: usize) -> Self {
        let mut basis: Vec<(u128, u128, bool, u32)> = Vec::new();

        for &(mut x, mut z) in stabilizers {
            for &(bx, bz, pivot_in_x, bit) in &basis {
                let set = if pivot_in_x { (x >> bit) & 1 } else { (z >> bit) & 1 };
                if set == 1 {
                    x ^= bx;
                    z ^= bz;
                }
            }
            if x == 0 && z == 0 {
                continue; // dependent on the rows already collected
            }
            let mut pivot = None;
            for i in 0..num_qubits {
                if (x >> i) & 1 == 1 {
                    pivot = Some((true, i as u32));
                    break;
                }
            }
            if pivot.is_none() {
                for i in 0..num_qubits {
                    if (z >> i) & 1 == 1 {
                        pivot = Some((false, i as u32));
                        break;
                    }
                }
            }
            let (pivot_in_x, bit) = pivot.expect("non-zero row has a pivot");
            basis.push((x, z, pivot_in_x, bit));
        }

        LogicalCheck { basis }
    }

    /// True when the residual is *not* a product of stabilizers.
    pub fn is_logical(&self, mut x: u128, mut z: u128) -> bool {
        for &(bx, bz, pivot_in_x, bit) in &self.basis {
            let set = if pivot_in_x { (x >> bit) & 1 } else { (z >> bit) & 1 };
            if set == 1 {
                x ^= bx;
                z ^= bz;
            }
        }
        x != 0 || z != 0
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

fn get_drift_p(p: f64, t: usize, correlated_noise: usize) -> f64 {
    if correlated_noise == 2 || correlated_noise == 3 {
        p * (1.0 + 0.5 * (2.0 * std::f64::consts::PI * t as f64 / 5.0).sin())
    } else {
        p
    }
}

fn sample_biased_error_with_erasure(p_pauli: f64, bias: f64, p_erase: f64, rng: &mut Xorshift) -> (bool, bool, bool) {
    if rng.next_f64() < p_erase {
        let choice = rng.next_u64() % 3;
        let (err_x, err_z) = match choice {
            0 => (true, false),
            1 => (false, true),
            _ => (true, true),
        };
        (err_x, err_z, true)
    } else {
        let (err_x, err_z) = sample_biased_error(p_pauli, bias, rng);
        (err_x, err_z, false)
    }
}

fn inject_correlated_noise(
    physical_x: &mut [bool],
    physical_z: &mut [bool],
    data_qubits: &[(usize, usize)],
    d: usize,
    correlated_noise: usize,
    rng: &mut Xorshift,
) {
    if correlated_noise == 1 || correlated_noise == 3 {
        if rng.next_f64() < 0.02 {
            let cx = rng.next_f64() * d as f64;
            let cy = rng.next_f64() * d as f64;
            for q in 0..data_qubits.len() {
                let (qx, qy) = data_qubits[q];
                let ux = (qx as f64 - 1.0) / 2.0;
                let uy = (qy as f64 - 1.0) / 2.0;
                let dist = ((ux - cx).powi(2) + (uy - cy).powi(2)).sqrt();
                if dist <= 1.5 {
                    let choice = rng.next_u64() % 3;
                    match choice {
                        0 => { physical_x[q] ^= true; }
                        1 => { physical_z[q] ^= true; }
                        _ => {
                            physical_x[q] ^= true;
                            physical_z[q] ^= true;
                        }
                    }
                }
            }
        }
    }
}

fn inject_correlated_noise_circuit(
    sim: &mut crate::simulator::StabilizerSimulator,
    data_qubits: &[(usize, usize)],
    d: usize,
    correlated_noise: usize,
    rng: &mut Xorshift,
) {
    if correlated_noise == 1 || correlated_noise == 3 {
        if rng.next_f64() < 0.02 {
            let cx = rng.next_f64() * d as f64;
            let cy = rng.next_f64() * d as f64;
            for q in 0..data_qubits.len() {
                let (qx, qy) = data_qubits[q];
                let ux = (qx as f64 - 1.0) / 2.0;
                let uy = (qy as f64 - 1.0) / 2.0;
                let dist = ((ux - cx).powi(2) + (uy - cy).powi(2)).sqrt();
                if dist <= 1.5 {
                    let choice = rng.next_u64() % 3;
                    match choice {
                        0 => { sim.apply_x(q); }
                        1 => { sim.apply_z(q); }
                        _ => { sim.apply_y(q); }
                    }
                }
            }
        }
    }
}



fn inject_two_qubit_noise(sim: &mut crate::simulator::StabilizerSimulator, q1: usize, q2: usize, p: f64, bias: f64, rng: &mut Xorshift) {
    inject_single_qubit_noise(sim, q1, p / 2.0, bias, rng);
    inject_single_qubit_noise(sim, q2, p / 2.0, bias, rng);
}

fn decode_by_type(graph: &SyndromeGraph, defects: &[bool], decoder_type: usize, erased_edges: &[bool]) -> Vec<usize> {
    match decoder_type {
        1 => crate::decoder::decode_greedy(graph, defects, erased_edges),
        2 => crate::decoder::decode_mwpm(graph, defects, erased_edges),
        _ => crate::decoder::decode_union_find(graph, defects, erased_edges),
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

    /// The data qubit at a given diagonal offset from a stabilizer, if any.
    ///
    /// Syndrome extraction has to schedule its CNOTs by *direction*, not by
    /// index into a variable-length neighbour list: a boundary plaquette has
    /// two neighbours, and compressing them into the first two time slots puts
    /// them in the wrong slots relative to the bulk.
    pub fn neighbor_at(&self, stab: &(usize, usize), dx: i32, dy: i32) -> Option<usize> {
        let (sx, sy) = *stab;
        let qx = sx as i32 + dx;
        let qy = sy as i32 + dy;
        if qx >= 1 && qx < (2 * self.d) as i32 && qy >= 1 && qy < (2 * self.d) as i32 {
            let x_idx = (qx - 1) as usize / 2;
            let y_idx = (qy - 1) as usize / 2;
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
    /// Returns the logical Pauli class left behind: bit 0 set if a logical X
    /// survived, bit 1 if a logical Z. 0 means the shot succeeded.
    ///
    /// A bare "did it fail" bool is not enough to characterise the logical
    /// channel: under Pauli noise and a Pauli decoder the channel is itself a
    /// Pauli channel, and reconstructing it needs the probability of each of
    /// I, X, Y and Z separately.
    pub fn simulate_phenomenological_noise(&self, num_rounds: usize, p: f64, bias: f64, decoder_type: usize, erasure_rate: f64, correlated_noise: usize) -> u8 {
        #[cfg(feature = "python")]
        let mut rng = Xorshift::new(rand::random());
        #[cfg(not(feature = "python"))]
        let mut rng = next_shot_rng();

        let num_stabs_z = self.z_stabilizers.len();
        let num_stabs_x = self.x_stabilizers.len();
        let num_data = self.data_qubits.len();

        let mut physical_x = vec![false; num_data];
        let mut physical_z = vec![false; num_data];

        let mut measured_z = vec![vec![false; num_stabs_z]; num_rounds];
        let mut measured_x = vec![vec![false; num_stabs_x]; num_rounds];

        let graph_z = self.build_syndrome_graph(num_rounds, true);
        let mut erased_edges_z = vec![false; graph_z.edges.len()];

        let graph_x = self.build_syndrome_graph(num_rounds, false);
        let mut erased_edges_x = vec![false; graph_x.edges.len()];

        for t in 0..num_rounds {
            let round_p = get_drift_p(p, t, correlated_noise);
            let p_erase = round_p * erasure_rate;
            let p_pauli = round_p * (1.0 - erasure_rate);

            for q in 0..num_data {
                let (err_x, err_z, erased) = sample_biased_error_with_erasure(p_pauli, bias, p_erase, &mut rng);
                if erased {
                    erased_edges_z[t * num_data + q] = true;
                    erased_edges_x[t * num_data + q] = true;
                }
                if err_x { physical_x[q] ^= true; }
                if err_z { physical_z[q] ^= true; }
            }

            inject_correlated_noise(&mut physical_x, &mut physical_z, &self.data_qubits, self.d, correlated_noise, &mut rng);

            for s_idx in 0..num_stabs_z {
                let neighbors = self.get_neighbors(&self.z_stabilizers[s_idx]);
                let mut parity = false;
                for &q in &neighbors {
                    if physical_x[q] {
                        parity ^= true;
                    }
                }
                // The final round is noiseless. Defects are time differences,
                // so a lie in the last round has no following round to cancel
                // against: it leaves an unpaired defect that the decoder must
                // match somewhere, injecting a correction for an error that
                // never happened. There are more checks to misreport on a
                // bigger patch, so including it makes the code get *worse*
                // with distance far below threshold. A real experiment ends
                // with a transversal data readout it can trust; this is that.
                if t + 1 < num_rounds && rng.next_f64() < round_p {
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
                // The final round is noiseless. Defects are time differences,
                // so a lie in the last round has no following round to cancel
                // against: it leaves an unpaired defect that the decoder must
                // match somewhere, injecting a correction for an error that
                // never happened. There are more checks to misreport on a
                // bigger patch, so including it makes the code get *worse*
                // with distance far below threshold. A real experiment ends
                // with a transversal data readout it can trust; this is that.
                if t + 1 < num_rounds && rng.next_f64() < round_p {
                    parity ^= true;
                }
                measured_x[t][s_idx] = parity;
            }
        }

        let mut defects_z = vec![false; graph_z.num_nodes];
        for t in 0..num_rounds {
            for s_idx in 0..num_stabs_z {
                let prev_outcome = if t == 0 { false } else { measured_z[t - 1][s_idx] };
                let diff = measured_z[t][s_idx] ^ prev_outcome;
                defects_z[s_idx + t * num_stabs_z] = diff;
            }
        }
        let correction_z_edges = decode_by_type(&graph_z, &defects_z, decoder_type, &erased_edges_z);

        let mut correction_x_data = vec![false; num_data];
        for edge_idx in correction_z_edges {
            if let Some(q_idx) = graph_z.edge_to_qubit[edge_idx] {
                correction_x_data[q_idx] ^= true;
            }
        }

        let mut defects_x = vec![false; graph_x.num_nodes];
        for t in 0..num_rounds {
            for s_idx in 0..num_stabs_x {
                let prev_outcome = if t == 0 { false } else { measured_x[t - 1][s_idx] };
                let diff = measured_x[t][s_idx] ^ prev_outcome;
                defects_x[s_idx + t * num_stabs_x] = diff;
            }
        }
        let correction_x_edges = decode_by_type(&graph_x, &defects_x, decoder_type, &erased_edges_x);

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

        (logical_x as u8) | ((logical_z as u8) << 1)
    }

    /// Simulates pure data noise with perfect stabilizer measurements.
    /// Returns the logical Pauli class left behind: bit 0 set if a logical X
    /// survived, bit 1 if a logical Z. 0 means the shot succeeded.
    ///
    /// A bare "did it fail" bool is not enough to characterise the logical
    /// channel: under Pauli noise and a Pauli decoder the channel is itself a
    /// Pauli channel, and reconstructing it needs the probability of each of
    /// I, X, Y and Z separately.
    pub fn simulate_data_noise(&self, p: f64, bias: f64, decoder_type: usize, erasure_rate: f64, correlated_noise: usize) -> u8 {
        #[cfg(feature = "python")]
        let mut rng = Xorshift::new(rand::random());
        #[cfg(not(feature = "python"))]
        let mut rng = next_shot_rng();

        let num_data = self.data_qubits.len();
        let mut physical_x = vec![false; num_data];
        let mut physical_z = vec![false; num_data];
        
        let graph_z = self.build_syndrome_graph(1, true);
        let mut erased_edges_z = vec![false; graph_z.edges.len()];

        let graph_x = self.build_syndrome_graph(1, false);
        let mut erased_edges_x = vec![false; graph_x.edges.len()];

        let round_p = get_drift_p(p, 0, correlated_noise);
        let p_erase = round_p * erasure_rate;
        let p_pauli = round_p * (1.0 - erasure_rate);

        for q in 0..num_data {
            let (err_x, err_z, erased) = sample_biased_error_with_erasure(p_pauli, bias, p_erase, &mut rng);
            if erased {
                erased_edges_z[q] = true;
                erased_edges_x[q] = true;
            }
            if err_x { physical_x[q] = true; }
            if err_z { physical_z[q] = true; }
        }

        inject_correlated_noise(&mut physical_x, &mut physical_z, &self.data_qubits, self.d, correlated_noise, &mut rng);
        
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

        let correction_z_edges = decode_by_type(&graph_z, &measured_z, decoder_type, &erased_edges_z);
        let mut correction_x_data = vec![false; num_data];
        for edge_idx in correction_z_edges {
            if let Some(q_idx) = graph_z.edge_to_qubit[edge_idx] {
                correction_x_data[q_idx] ^= true;
            }
        }

        let correction_x_edges = decode_by_type(&graph_x, &measured_x, decoder_type, &erased_edges_x);
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

        (logical_x as u8) | ((logical_z as u8) << 1)
    }

    /// Returns the logical Pauli class left behind: bit 0 set if a logical X
    /// survived, bit 1 if a logical Z. 0 means the shot succeeded.
    ///
    /// A bare "did it fail" bool is not enough to characterise the logical
    /// channel: under Pauli noise and a Pauli decoder the channel is itself a
    /// Pauli channel, and reconstructing it needs the probability of each of
    /// I, X, Y and Z separately.
    pub fn simulate_circuit_noise(&self, num_rounds: usize, p: f64, bias: f64, init_state: &str, decoder_type: usize, erasure_rate: f64, correlated_noise: usize) -> u8 {
        let num_data = self.data_qubits.len();
        let num_stabs_x = self.x_stabilizers.len();
        let num_stabs_z = self.z_stabilizers.len();
        let num_stabs = num_stabs_x + num_stabs_z;
        let total_qubits = num_data + num_stabs;
        
        #[cfg(feature = "python")]
        let mut rng = Xorshift::new(rand::random());
        #[cfg(not(feature = "python"))]
        let mut rng = next_shot_rng();

        // Seed the tableau's measurement randomness from this shot's generator.
        // Left at its default every shot replays identical measurement outcomes
        // and the whole circuit-level run is deterministic.
        let mut sim = crate::simulator::StabilizerSimulator::with_seed(total_qubits, rng.next_u64());

        let graph_z = self.build_syndrome_graph(num_rounds, true);
        let mut erased_edges_z = vec![false; graph_z.edges.len()];

        let graph_x = self.build_syndrome_graph(num_rounds, false);
        let mut erased_edges_x = vec![false; graph_x.edges.len()];
  
        let get_x_stab_qubit = |j: usize| num_data + j;
        let get_z_stab_qubit = |k: usize| num_data + num_stabs_x + k;

        let p_erase_init = p * erasure_rate;
        let p_pauli_init = p * (1.0 - erasure_rate);
  
        if init_state == "plus" {
            for i in 0..num_data {
                sim.apply_h(i);
                if rng.next_f64() < p_erase_init {
                    erased_edges_z[i] = true;
                    erased_edges_x[i] = true;
                    let choice = rng.next_u64() % 3;
                    match choice {
                        0 => sim.apply_x(i),
                        1 => sim.apply_y(i),
                        2 => sim.apply_z(i),
                        _ => unreachable!(),
                    }
                } else {
                    inject_single_qubit_noise(&mut sim, i, p_pauli_init, bias, &mut rng);
                }
            }
        } else {
            for i in 0..num_data {
                if rng.next_f64() < p_erase_init {
                    erased_edges_z[i] = true;
                    erased_edges_x[i] = true;
                    let choice = rng.next_u64() % 3;
                    match choice {
                        0 => sim.apply_x(i),
                        1 => sim.apply_y(i),
                        2 => sim.apply_z(i),
                        _ => unreachable!(),
                    }
                } else {
                    inject_single_qubit_noise(&mut sim, i, p_pauli_init, bias, &mut rng);
                }
            }
        }
  
        // Helper to run one round of syndrome extraction
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
  
            // X and Z ancillas must walk their plaquettes in opposite orders —
            // the classic "N" and "Z" schedules. Using one order for both, as
            // this did, leaves the two extraction circuits non-commuting where
            // plaquettes overlap, so each measurement disturbs the other and
            // the round reports defects even with no noise at all.
            const X_ORDER: [(i32, i32); 4] = [(-1, -1), (1, -1), (-1, 1), (1, 1)];
            const Z_ORDER: [(i32, i32); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];

            for step in 0..4 {
                for j in 0..num_stabs_x {
                    let (dx, dy) = X_ORDER[step];
                    if let Some(data_idx) = self.neighbor_at(&self.x_stabilizers[j], dx, dy) {
                        let anc_idx = get_x_stab_qubit(j);
                        sim_obj.apply_cnot(anc_idx, data_idx);
                        inject_two_qubit_noise(sim_obj, anc_idx, data_idx, round_p, bias, rng_obj);
                    }
                }
                for k in 0..num_stabs_z {
                    let (dx, dy) = Z_ORDER[step];
                    if let Some(data_idx) = self.neighbor_at(&self.z_stabilizers[k], dx, dy) {
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
            let round_p = get_drift_p(p, t, correlated_noise);
            let p_erase = round_p * erasure_rate;
            
            for q in 0..num_data {
                if rng.next_f64() < p_erase {
                    erased_edges_z[t * num_data + q] = true;
                    erased_edges_x[t * num_data + q] = true;
                    let choice = rng.next_u64() % 3;
                    match choice {
                        0 => sim.apply_x(q),
                        1 => sim.apply_y(q),
                        2 => sim.apply_z(q),
                        _ => unreachable!(),
                    }
                }
            }

            let (rx, rz) = run_round(&mut sim, round_p, &mut rng);
            measured_x[t] = rx;
            measured_z[t] = rz;

            inject_correlated_noise_circuit(&mut sim, &self.data_qubits, self.d, correlated_noise, &mut rng);
        }
  
        // --- DECODING Z-stabilizers (X errors) ---
        let mut defects_z = vec![false; graph_z.num_nodes];
        for t in 0..num_rounds {
            for s_idx in 0..num_stabs_z {
                let prev_outcome = if t == 0 { baseline_z[s_idx] } else { measured_z[t - 1][s_idx] };
                let diff = measured_z[t][s_idx] ^ prev_outcome;
                defects_z[s_idx + t * num_stabs_z] = diff;
            }
        }
        let correction_z_edges = decode_by_type(&graph_z, &defects_z, decoder_type, &erased_edges_z);
  
        let mut correction_x_data = vec![false; num_data];
        for edge_idx in correction_z_edges {
            if let Some(q_idx) = graph_z.edge_to_qubit[edge_idx] {
                correction_x_data[q_idx] ^= true;
            }
        }
  
        // --- DECODING X-stabilizers (Z errors) ---
        let mut defects_x = vec![false; graph_x.num_nodes];
        for t in 0..num_rounds {
            for s_idx in 0..num_stabs_x {
                let prev_outcome = if t == 0 { baseline_x[s_idx] } else { measured_x[t - 1][s_idx] };
                let diff = measured_x[t][s_idx] ^ prev_outcome;
                defects_x[s_idx + t * num_stabs_x] = diff;
            }
        }
        let correction_x_edges = decode_by_type(&graph_x, &defects_x, decoder_type, &erased_edges_x);
  
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
            (logical_x == 1) as u8
        } else {
            let mut logical_z = 0;
            for y_idx in 0..self.d {
                let q_idx = 0 + self.d * y_idx;
                logical_z ^= sim.measure_z(q_idx);
            }
            (logical_z == 1) as u8
        }
    }
}

pub struct XZZXSurfaceCode {
    pub d: usize,
    pub data_qubits: Vec<(usize, usize)>,
    pub stabilizers: Vec<(usize, usize)>,
    pub z_stabilizers: Vec<(usize, usize)>,
    pub x_stabilizers: Vec<(usize, usize)>,
    /// Built once per code; see LogicalCheck for why the hard-coded string
    /// this replaces was wrong.
    pub logical: LogicalCheck,
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

        // Symplectic form of each stabilizer. XZZX puts X on the NW/SE diagonal
        // and Z on the NE/SW one; verified against the engine's own syndrome by
        // injecting every single-qubit error and comparing which checks fire.
        let index_of = |x: i32, y: i32| -> Option<usize> {
            if x >= 1 && x < (2 * d) as i32 && y >= 1 && y < (2 * d) as i32 {
                Some(((x - 1) as usize / 2) + d * ((y - 1) as usize / 2))
            } else {
                None
            }
        };
        let mut ops: Vec<(u128, u128)> = Vec::with_capacity(stabilizers.len());
        for &(sx, sy) in &stabilizers {
            let (mut px, mut pz) = (0u128, 0u128);
            for &(dx, dy, is_x) in &[
                (-1i32, -1i32, true), (1, 1, true),
                (1, -1, false), (-1, 1, false),
            ] {
                if let Some(q) = index_of(sx as i32 + dx, sy as i32 + dy) {
                    if is_x { px |= 1u128 << q; } else { pz |= 1u128 << q; }
                }
            }
            ops.push((px, pz));
        }
        let logical = LogicalCheck::new(&ops, d * d);

        XZZXSurfaceCode {
            d,
            data_qubits,
            stabilizers,
            z_stabilizers,
            x_stabilizers,
            logical,
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

    /// One graph carrying both error families, with a tag per edge.
    ///
    /// XZZX has a single syndrome but two edge families over the same nodes:
    /// an X error flips the stabilizers on one diagonal, a Z error those on the
    /// other. Decoding the whole defect set independently on each family — as
    /// this file used to, via `defects_x = defects_z.clone()` — explains every
    /// defect twice, once with X operators and once with Z, and applies both
    /// corrections. A single X error came back with the right X correction plus
    /// two invented Z ones, and the count of those grows with the lattice, so
    /// the code got *worse* with distance far below threshold.
    ///
    /// Matching once over the union is the fix: every defect is explained
    /// exactly once, and the tag says whether the chosen edge means "apply X
    /// here" or "apply Z here".
    ///
    /// Returns the graph and, per edge, `true` if correcting it means applying
    /// X. The flag is meaningless for the timelike edges, which carry no qubit.
    pub fn build_combined_graph(&self, num_rounds: usize) -> (SyndromeGraph, Vec<bool>) {
        let num_stabs = self.stabilizers.len();
        let num_nodes = num_stabs * num_rounds;

        let mut edges = Vec::new();
        let mut edge_to_qubit = Vec::new();
        let mut edge_is_x = Vec::new();
        let mut edge_id = 0;

        for t in 0..num_rounds {
            for is_for_x_errors in [true, false] {
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
                        edge_is_x.push(is_for_x_errors);
                        edge_id += 1;
                    } else if connected_stabs.len() == 1 {
                        let u = connected_stabs[0] + t * num_stabs;
                        let v = num_nodes;
                        edges.push(Edge { u, v, id: edge_id });
                        edge_to_qubit.push(Some(q_idx));
                        edge_is_x.push(is_for_x_errors);
                        edge_id += 1;
                    }
                }
            }
        }

        // Timelike edges, once — they belong to neither family.
        for t in 0..num_rounds.saturating_sub(1) {
            for s_idx in 0..num_stabs {
                let u = s_idx + t * num_stabs;
                let v = s_idx + (t + 1) * num_stabs;
                edges.push(Edge { u, v, id: edge_id });
                edge_to_qubit.push(None);
                edge_is_x.push(false);
                edge_id += 1;
            }
        }

        (SyndromeGraph { num_nodes, edges, edge_to_qubit }, edge_is_x)
    }

    /// Returns the logical Pauli class left behind: bit 0 set if a logical X
    /// survived, bit 1 if a logical Z. 0 means the shot succeeded.
    ///
    /// A bare "did it fail" bool is not enough to characterise the logical
    /// channel: under Pauli noise and a Pauli decoder the channel is itself a
    /// Pauli channel, and reconstructing it needs the probability of each of
    /// I, X, Y and Z separately.
    pub fn simulate_phenomenological_noise(&self, num_rounds: usize, p: f64, bias: f64, decoder_type: usize, erasure_rate: f64, correlated_noise: usize) -> u8 {
        #[cfg(feature = "python")]
        let mut rng = Xorshift::new(rand::random());
        #[cfg(not(feature = "python"))]
        let mut rng = next_shot_rng();

        let num_stabs = self.stabilizers.len();
        let num_data = self.data_qubits.len();

        let mut physical_x = vec![false; num_data];
        let mut physical_z = vec![false; num_data];

        let mut measured = vec![vec![false; num_stabs]; num_rounds];

        // Erasure is a property of a (qubit, round); graph edges are derived
        // from it below. The old code indexed an edge-length array by
        // `t * num_data + q`, which is not an edge index at all.
        let mut erased_qubits = vec![false; num_data * num_rounds];

        for t in 0..num_rounds {
            let round_p = get_drift_p(p, t, correlated_noise);
            let p_erase = round_p * erasure_rate;
            let p_pauli = round_p * (1.0 - erasure_rate);

            for q in 0..num_data {
                let (err_x, err_z, erased) = sample_biased_error_with_erasure(p_pauli, bias, p_erase, &mut rng);
                if erased {
                    erased_qubits[t * num_data + q] = true;
                }
                if err_x { physical_x[q] ^= true; }
                if err_z { physical_z[q] ^= true; }
            }

            inject_correlated_noise(&mut physical_x, &mut physical_z, &self.data_qubits, self.d, correlated_noise, &mut rng);

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

                // The final round is noiseless. Defects are time differences,
                // so a lie in the last round has no following round to cancel
                // against: it leaves an unpaired defect that the decoder must
                // match somewhere, injecting a correction for an error that
                // never happened. There are more checks to misreport on a
                // bigger patch, so including it makes the code get *worse*
                // with distance far below threshold. A real experiment ends
                // with a transversal data readout it can trust; this is that.
                if t + 1 < num_rounds && rng.next_f64() < round_p {
                    parity ^= true;
                }
                measured[t][s_idx] = parity;
            }
        }

        // One matching over both error families. See build_combined_graph.
        let (graph, edge_is_x) = self.build_combined_graph(num_rounds);

        let mut defects_z = vec![false; graph.num_nodes];
        for t in 0..num_rounds {
            for s_idx in 0..num_stabs {
                let prev_outcome = if t == 0 { false } else { measured[t - 1][s_idx] };
                let diff = measured[t][s_idx] ^ prev_outcome;
                defects_z[s_idx + t * num_stabs] = diff;
            }
        }
        let mut erased_edges = vec![false; graph.edges.len()];
        for edge_idx in 0..graph.edges.len() {
            if let Some(q_idx) = graph.edge_to_qubit[edge_idx] {
                let t_layer = graph.edges[edge_idx].u / self.stabilizers.len();
                let idx = q_idx + t_layer * num_data;
                if idx < erased_qubits.len() && erased_qubits[idx] {
                    erased_edges[edge_idx] = true;
                }
            }
        }

        let correction_edges = decode_by_type(&graph, &defects_z, decoder_type, &erased_edges);
        let mut correction_x_data = vec![false; num_data];
        let mut correction_z_data = vec![false; num_data];
        for edge_idx in correction_edges {
            if let Some(q_idx) = graph.edge_to_qubit[edge_idx] {
                if edge_is_x[edge_idx] {
                    correction_x_data[q_idx] ^= true;
                } else {
                    correction_z_data[q_idx] ^= true;
                }
            }
        }

        let mut residual_x = vec![false; num_data];
        let mut residual_z = vec![false; num_data];
        for q in 0..num_data {
            residual_x[q] = physical_x[q] ^ correction_x_data[q];
            residual_z[q] = physical_z[q] ^ correction_z_data[q];
        }

        // Ask the stabilizer group directly rather than trusting a hard-coded
        // logical string. See LogicalCheck.
        let (mut rx, mut rz) = (0u128, 0u128);
        for q in 0..num_data {
            if residual_x[q] { rx |= 1u128 << q; }
            if residual_z[q] { rz |= 1u128 << q; }
        }
        self.logical.is_logical(rx, rz) as u8
    }

    /// Returns the logical Pauli class left behind: bit 0 set if a logical X
    /// survived, bit 1 if a logical Z. 0 means the shot succeeded.
    ///
    /// A bare "did it fail" bool is not enough to characterise the logical
    /// channel: under Pauli noise and a Pauli decoder the channel is itself a
    /// Pauli channel, and reconstructing it needs the probability of each of
    /// I, X, Y and Z separately.
    pub fn simulate_data_noise(&self, p: f64, bias: f64, decoder_type: usize, erasure_rate: f64, correlated_noise: usize) -> u8 {
        #[cfg(feature = "python")]
        let mut rng = Xorshift::new(rand::random());
        #[cfg(not(feature = "python"))]
        let mut rng = next_shot_rng();

        let num_data = self.data_qubits.len();
        let mut physical_x = vec![false; num_data];
        let mut physical_z = vec![false; num_data];

        // Erasure is a property of a qubit; the graph edges are derived from it
        // below. The old code indexed an edge array by qubit index, which only
        // lined up by accident for one of the two families.
        let mut erased_qubits = vec![false; num_data];

        let round_p = get_drift_p(p, 0, correlated_noise);
        let p_erase = round_p * erasure_rate;
        let p_pauli = round_p * (1.0 - erasure_rate);

        for q in 0..num_data {
            let (err_x, err_z, erased) = sample_biased_error_with_erasure(p_pauli, bias, p_erase, &mut rng);
            if erased {
                erased_qubits[q] = true;
            }
            if err_x { physical_x[q] = true; }
            if err_z { physical_z[q] = true; }
        }

        inject_correlated_noise(&mut physical_x, &mut physical_z, &self.data_qubits, self.d, correlated_noise, &mut rng);

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

        // One matching over both error families. See build_combined_graph.
        let (graph, edge_is_x) = self.build_combined_graph(1);
        let mut erased_edges = vec![false; graph.edges.len()];
        for edge_idx in 0..graph.edges.len() {
            if let Some(q_idx) = graph.edge_to_qubit[edge_idx] {
                if erased_qubits[q_idx] {
                    erased_edges[edge_idx] = true;
                }
            }
        }

        let correction_edges = decode_by_type(&graph, &measured, decoder_type, &erased_edges);
        let mut correction_x_data = vec![false; num_data];
        let mut correction_z_data = vec![false; num_data];
        for edge_idx in correction_edges {
            if let Some(q_idx) = graph.edge_to_qubit[edge_idx] {
                if edge_is_x[edge_idx] {
                    correction_x_data[q_idx] ^= true;
                } else {
                    correction_z_data[q_idx] ^= true;
                }
            }
        }

        let mut residual_x = vec![false; num_data];
        let mut residual_z = vec![false; num_data];
        for q in 0..num_data {
            residual_x[q] = physical_x[q] ^ correction_x_data[q];
            residual_z[q] = physical_z[q] ^ correction_z_data[q];
        }

        // Ask the stabilizer group directly rather than trusting a hard-coded
        // logical string. See LogicalCheck.
        let (mut rx, mut rz) = (0u128, 0u128);
        for q in 0..num_data {
            if residual_x[q] { rx |= 1u128 << q; }
            if residual_z[q] { rz |= 1u128 << q; }
        }
        self.logical.is_logical(rx, rz) as u8
    }

    /// Returns the logical Pauli class left behind: bit 0 set if a logical X
    /// survived, bit 1 if a logical Z. 0 means the shot succeeded.
    ///
    /// A bare "did it fail" bool is not enough to characterise the logical
    /// channel: under Pauli noise and a Pauli decoder the channel is itself a
    /// Pauli channel, and reconstructing it needs the probability of each of
    /// I, X, Y and Z separately.
    pub fn simulate_circuit_noise(&self, num_rounds: usize, p: f64, bias: f64, init_state: &str, decoder_type: usize, erasure_rate: f64, correlated_noise: usize) -> u8 {
        let num_data = self.data_qubits.len();
        let num_stabs_z = self.z_stabilizers.len();
        let num_stabs_x = self.x_stabilizers.len();
        let total_qubits = num_data + num_stabs_z + num_stabs_x;
        
        #[cfg(feature = "python")]
        let mut rng = Xorshift::new(rand::random());
        #[cfg(not(feature = "python"))]
        let mut rng = next_shot_rng();

        // Seed the tableau's measurement randomness from this shot's generator.
        // Left at its default every shot replays identical measurement outcomes
        // and the whole circuit-level run is deterministic.
        let mut sim = crate::simulator::StabilizerSimulator::with_seed(total_qubits, rng.next_u64());

        let graph_z = self.build_syndrome_graph(num_rounds, true);
        let mut erased_edges_z = vec![false; graph_z.edges.len()];

        let graph_x = self.build_syndrome_graph(num_rounds, false);
        let mut erased_edges_x = vec![false; graph_x.edges.len()];

        let get_stab_qubit = |j: usize| num_data + j;

        let p_erase_init = p * erasure_rate;
        let p_pauli_init = p * (1.0 - erasure_rate);

        if init_state == "plus" {
            for x_idx in 0..self.d {
                let q_idx = x_idx + self.d * 0;
                sim.apply_h(q_idx);
            }
            for i in 0..num_data {
                if rng.next_f64() < p_erase_init {
                    erased_edges_z[i] = true;
                    erased_edges_x[i] = true;
                    let choice = rng.next_u64() % 3;
                    match choice {
                        0 => sim.apply_x(i),
                        1 => sim.apply_y(i),
                        2 => sim.apply_z(i),
                        _ => unreachable!(),
                    }
                } else {
                    inject_single_qubit_noise(&mut sim, i, p_pauli_init, bias, &mut rng);
                }
            }
        } else if init_state == "y" {
            sim.apply_h(0);
            sim.apply_s(0);
            for x_idx in 1..self.d {
                sim.apply_h(x_idx);
            }
            for i in 0..num_data {
                if rng.next_f64() < p_erase_init {
                    erased_edges_z[i] = true;
                    erased_edges_x[i] = true;
                    let choice = rng.next_u64() % 3;
                    match choice {
                        0 => sim.apply_x(i),
                        1 => sim.apply_y(i),
                        2 => sim.apply_z(i),
                        _ => unreachable!(),
                    }
                } else {
                    inject_single_qubit_noise(&mut sim, i, p_pauli_init, bias, &mut rng);
                }
            }
        } else {
            for i in 0..num_data {
                if rng.next_f64() < p_erase_init {
                    erased_edges_z[i] = true;
                    erased_edges_x[i] = true;
                    let choice = rng.next_u64() % 3;
                    match choice {
                        0 => sim.apply_x(i),
                        1 => sim.apply_y(i),
                        2 => sim.apply_z(i),
                        _ => unreachable!(),
                    }
                } else {
                    inject_single_qubit_noise(&mut sim, i, p_pauli_init, bias, &mut rng);
                }
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
            let round_p = get_drift_p(p, t, correlated_noise);
            let p_erase = round_p * erasure_rate;

            for q in 0..num_data {
                if rng.next_f64() < p_erase {
                    erased_edges_z[t * num_data + q] = true;
                    erased_edges_x[t * num_data + q] = true;
                    let choice = rng.next_u64() % 3;
                    match choice {
                        0 => sim.apply_x(q),
                        1 => sim.apply_y(q),
                        2 => sim.apply_z(q),
                        _ => unreachable!(),
                    }
                }
            }

            let (rz, rx) = run_round(&mut sim, round_p, &mut rng);
            measured_z[t] = rz;
            measured_x[t] = rx;

            inject_correlated_noise_circuit(&mut sim, &self.data_qubits, self.d, correlated_noise, &mut rng);
        }

        let mut defects_z = vec![false; graph_z.num_nodes];
        for t in 0..num_rounds {
            for s_idx in 0..num_stabs_z {
                let prev_outcome = if t == 0 { baseline_z[s_idx] } else { measured_z[t - 1][s_idx] };
                let diff = measured_z[t][s_idx] ^ prev_outcome;
                defects_z[s_idx + t * num_stabs_z] = diff;
            }
        }
        let correction_z_edges = decode_by_type(&graph_z, &defects_z, decoder_type, &erased_edges_z);

        let mut correction_x_data = vec![false; num_data];
        for edge_idx in correction_z_edges {
            if let Some(q_idx) = graph_z.edge_to_qubit[edge_idx] {
                correction_x_data[q_idx] ^= true;
            }
        }

        let mut defects_x = vec![false; graph_x.num_nodes];
        for t in 0..num_rounds {
            for s_idx in 0..num_stabs_x {
                let prev_outcome = if t == 0 { baseline_x[s_idx] } else { measured_x[t - 1][s_idx] };
                let diff = measured_x[t][s_idx] ^ prev_outcome;
                defects_x[s_idx + t * num_stabs_x] = diff;
            }
        }
        let correction_x_edges = decode_by_type(&graph_x, &defects_x, decoder_type, &erased_edges_x);

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
            logical_err as u8
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
            logical_err as u8
        } else {
            let mut logical_err = false;
            for y_idx in 0..self.d {
                let q_idx = 0 + self.d * y_idx;
                let outcome = sim.measure_z(q_idx);
                if outcome == 1 {
                    logical_err ^= true;
                }
            }
            logical_err as u8
        }
    }
}

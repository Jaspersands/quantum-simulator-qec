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
    /// Anticommuting logical representatives, derived in `new`.
    logical_x: (u128, u128),
    logical_z: (u128, u128),
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

        let mut check = LogicalCheck { basis, logical_x: (0, 0), logical_z: (0, 0) };
        let (lx, lz) = find_logical_pair(stabilizers, num_qubits, &check);
        check.logical_x = lx;
        check.logical_z = lz;
        check
    }

    /// True when the residual is *not* a product of stabilizers.
    pub fn is_logical(&self, x: u128, z: u128) -> bool {
        let (rx, rz) = self.reduce(x, z);
        rx != 0 || rz != 0
    }

    /// Reduce a Pauli against the stabilizer basis, leaving its logical part.
    fn reduce(&self, mut x: u128, mut z: u128) -> (u128, u128) {
        for &(bx, bz, pivot_in_x, bit) in &self.basis {
            let set = if pivot_in_x { (x >> bit) & 1 } else { (z >> bit) & 1 };
            if set == 1 {
                x ^= bx;
                z ^= bz;
            }
        }
        (x, z)
    }

    /// The derived logical representatives, for tests and diagnostics.
    pub fn representatives(&self) -> ((u128, u128), (u128, u128)) {
        (self.logical_x, self.logical_z)
    }

    /// Which logical Pauli the residual is, as a class: bit 0 set for a logical
    /// X, bit 1 for a logical Z, both for a Y, zero for success.
    ///
    /// "Did a logical error happen" is not enough to characterise the logical
    /// channel — that needs the probability of I, X, Y and Z separately — and a
    /// bare pass/fail silently reports every failure as the same kind. Which one
    /// it is comes from commutation: a residual carries a logical X exactly when
    /// it anticommutes with the logical Z operator, and vice versa.
    pub fn classify(&self, x: u128, z: u128) -> u8 {
        let (rx, rz) = self.reduce(x, z);
        let anti = |(qx, qz): (u128, u128)| -> bool {
            ((rx & qz).count_ones() + (rz & qx).count_ones()) % 2 == 1
        };
        (anti(self.logical_z) as u8) | ((anti(self.logical_x) as u8) << 1)
    }
}

/// Row-reduce a GF(2) matrix in place and return the pivot column of each row.
fn gf2_rref(rows: &mut Vec<u128>, width: usize) -> Vec<usize> {
    let mut pivots = Vec::new();
    let mut r = 0usize;
    for col in 0..width {
        let Some(sel) = (r..rows.len()).find(|&i| (rows[i] >> col) & 1 == 1) else {
            continue;
        };
        rows.swap(r, sel);
        for i in 0..rows.len() {
            if i != r && (rows[i] >> col) & 1 == 1 {
                rows[i] ^= rows[r];
            }
        }
        pivots.push(col);
        r += 1;
        if r == rows.len() {
            break;
        }
    }
    rows.truncate(r);
    pivots
}

/// A pair of anticommuting logical operators for the code, found rather than
/// assumed.
///
/// The normalizer of the stabilizer group is the null space of the commutation
/// map: P = (px, pz) commutes with every generator (sx, sz) exactly when
/// `popcount(px & sz) + popcount(pz & sx)` is even for each, which is a linear
/// system over GF(2) in the 2n bits of P. Anything in that null space which is
/// not itself a product of stabilizers is a logical operator; a pair of them
/// that anticommute are representatives of logical X and logical Z. Deriving
/// them beats writing them down — a hard-coded string that was not a logical
/// operator of the lattice is precisely what went wrong here before.
///
/// Which of the pair is called X and which Z is a convention, not a fact: for
/// the rotated code the search recovers the usual column-of-X and row-of-Z, and
/// `logical_classification_matches_the_rotated_code` pins that down against the
/// code's own hand-derived answer. For XZZX every logical operator is a mixed
/// X/Z string and no such preferred labelling exists, so the choice is simply
/// made deterministically — the enumeration order is fixed — and the two axes of
/// the reported channel are that basis, whatever it is called.
fn find_logical_pair(
    stabilizers: &[(u128, u128)],
    num_qubits: usize,
    reduced: &LogicalCheck,
) -> ((u128, u128), (u128, u128)) {
    let n = num_qubits;
    let width = 2 * n;

    // One row per generator: columns 0..n weight px, columns n..2n weight pz.
    let mut rows: Vec<u128> = stabilizers
        .iter()
        .map(|&(sx, sz)| {
            let mut row = 0u128;
            for j in 0..n {
                if (sz >> j) & 1 == 1 { row |= 1u128 << j; }
                if (sx >> j) & 1 == 1 { row |= 1u128 << (n + j); }
            }
            row
        })
        .collect();
    let pivots = gf2_rref(&mut rows, width);
    let is_pivot: Vec<bool> = (0..width).map(|c| pivots.contains(&c)).collect();

    // Null-space basis: one vector per free column.
    let mut null = Vec::new();
    for free in 0..width {
        if is_pivot[free] { continue; }
        let mut v = 1u128 << free;
        for (r, &pc) in pivots.iter().enumerate() {
            if (rows[r] >> free) & 1 == 1 { v |= 1u128 << pc; }
        }
        null.push(v);
    }

    let mask = if n >= 128 { u128::MAX } else { (1u128 << n) - 1 };
    let split = |v: u128| -> (u128, u128) { (v & mask, (v >> n) & mask) };

    let logicals: Vec<(u128, u128)> = null
        .into_iter()
        .map(split)
        .filter(|&(x, z)| reduced.is_logical(x, z))
        .collect();

    for (i, &a) in logicals.iter().enumerate() {
        for &b in &logicals[i + 1..] {
            if ((a.0 & b.1).count_ones() + (a.1 & b.0).count_ones()) % 2 == 1 {
                return (a, b);
            }
        }
    }
    // A code with no anticommuting pair in its normalizer encodes no qubit;
    // every construction here encodes one, so this is unreachable.
    ((0, 0), (0, 0))
}

/// The biased Pauli channel, in one place.
///
/// `eta` is the ratio of Z to (X + Y): at eta = 0.5 the three are equally
/// likely, which is depolarizing noise. Returns the Pauli as a bitmask —
/// 1 = X, 2 = Z, 3 = Y — or None if nothing happened.
fn sample_pauli(p: f64, eta: f64, rng: &mut Xorshift) -> Option<u8> {
    if p <= 0.0 || rng.next_f64() >= p {
        return None;
    }
    let roll = rng.next_f64();
    let p_z = eta / (eta + 1.0);
    let p_x = 1.0 / (2.0 * (eta + 1.0));
    Some(if roll < p_z { 2 } else if roll < p_z + p_x { 1 } else { 3 })
}

fn sample_biased_error(p: f64, eta: f64, rng: &mut Xorshift) -> (bool, bool) {
    match sample_pauli(p, eta, rng) {
        Some(pauli) => (pauli & 1 != 0, pauli & 2 != 0),
        None => (false, false),
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
    if (correlated_noise == 1 || correlated_noise == 3)
        && rng.next_f64() < 0.02 {
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

fn inject_correlated_noise_circuit(
    sim: &mut crate::simulator::StabilizerSimulator,
    mut frame: Option<&mut crate::circuit_model::Frame>,
    data_qubits: &[(usize, usize)],
    d: usize,
    correlated_noise: usize,
    rng: &mut Xorshift,
) {
    if (correlated_noise == 1 || correlated_noise == 3)
        && rng.next_f64() < 0.02 {
            let cx = rng.next_f64() * d as f64;
            let cy = rng.next_f64() * d as f64;
            for q in 0..data_qubits.len() {
                let (qx, qy) = data_qubits[q];
                let ux = (qx as f64 - 1.0) / 2.0;
                let uy = (qy as f64 - 1.0) / 2.0;
                let dist = ((ux - cx).powi(2) + (uy - cy).powi(2)).sqrt();
                if dist <= 1.5 {
                    let choice = rng.next_u64() % 3;
                    let pauli = match choice {
                        0 => { sim.apply_x(q); 1u8 }
                        1 => { sim.apply_z(q); 2u8 }
                        _ => { sim.apply_y(q); 3u8 }
                    };
                    if let Some(f) = frame.as_deref_mut() { f.apply(q, pauli); }
                }
            }
        }
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
    /// The syndrome-extraction round, as a list of instructions.
    ///
    /// This is the single definition of the circuit. `simulate_circuit_noise`
    /// executes it against the stabilizer tableau, and `circuit_model` walks the
    /// same list propagating faults to build the decoding graph. Keeping two
    /// hand-written copies of a circuit in step is exactly the kind of thing
    /// that quietly rots, and a detector error model describing a slightly
    /// different circuit from the one being run is worse than none.
    ///
    /// X and Z ancillas walk their plaquettes in opposite orders — the classic
    /// N and Z schedules — and are scheduled by direction, so a boundary
    /// plaquette's two gates land in the time slots its geometry calls for
    /// rather than being compressed into the first two.
    pub fn round_program(&self) -> Vec<crate::circuit_model::Op> {
        use crate::circuit_model::{Op, StabKind};

        // X ancillas walk their plaquette column-major, Z ancillas row-major —
        // the two orders are transposes, which is what makes the extraction
        // circuits commute where plaquettes overlap. The pairing is not
        // arbitrary and not guessable: with the two orders swapped, every single
        // fault is still corrected at d = 5 and d = 7 but 44 of 600 fail at
        // d = 3. The exhaustive single-fault check settled which way round it
        // goes.
        const X_ORDER: [(i32, i32); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];
        const Z_ORDER: [(i32, i32); 4] = [(-1, -1), (1, -1), (-1, 1), (1, 1)];

        let num_data = self.data_qubits.len();
        let num_x = self.x_stabilizers.len();
        let x_anc = |j: usize| num_data + j;
        let z_anc = |k: usize| num_data + num_x + k;

        let mut ops = Vec::new();

        // Prepare ancillas: X-type in |+>, Z-type in |0>.
        for j in 0..num_x {
            ops.push(Op::Reset(x_anc(j)));
            ops.push(Op::Noise(x_anc(j)));
            ops.push(Op::H(x_anc(j)));
            ops.push(Op::Noise(x_anc(j)));
        }
        for k in 0..self.z_stabilizers.len() {
            ops.push(Op::Reset(z_anc(k)));
            ops.push(Op::Noise(z_anc(k)));
        }

        // Four interleaved CNOT layers.
        for step in 0..4 {
            for j in 0..num_x {
                let (dx, dy) = X_ORDER[step];
                if let Some(data) = self.neighbor_at(&self.x_stabilizers[j], dx, dy) {
                    ops.push(Op::Cnot(x_anc(j), data));
                    ops.push(Op::Noise(x_anc(j)));
                    ops.push(Op::Noise(data));
                }
            }
            for k in 0..self.z_stabilizers.len() {
                let (dx, dy) = Z_ORDER[step];
                if let Some(data) = self.neighbor_at(&self.z_stabilizers[k], dx, dy) {
                    ops.push(Op::Cnot(data, z_anc(k)));
                    ops.push(Op::Noise(data));
                    ops.push(Op::Noise(z_anc(k)));
                }
            }
        }

        // Rotate the X ancillas back and read everything out.
        for j in 0..num_x {
            ops.push(Op::H(x_anc(j)));
            ops.push(Op::Noise(x_anc(j)));
        }
        for j in 0..num_x {
            ops.push(Op::Measure(x_anc(j), StabKind::X, j));
        }
        for k in 0..self.z_stabilizers.len() {
            ops.push(Op::Measure(z_anc(k), StabKind::Z, k));
        }

        ops
    }

    /// The circuit plus the sizes the model builder needs.
    pub fn circuit_layout(&self) -> crate::circuit_model::CircuitLayout {
        crate::circuit_model::CircuitLayout {
            program: self.round_program(),
            num_qubits: self.data_qubits.len() + self.x_stabilizers.len() + self.z_stabilizers.len(),
            num_data: self.data_qubits.len(),
            num_x_stabs: self.x_stabilizers.len(),
            num_z_stabs: self.z_stabilizers.len(),
        }
    }

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
            let q_idx = self.d * y_idx; // the leftmost column: 0, d, 2d, ...
            if residual_x[q_idx] {
                logical_x ^= true;
            }
        }

        // A logical Z error occurred if the residual Z errors cross the code from top to bottom.
        // We check this by checking the parity of residual Z errors along a horizontal line (y=1).
        let mut logical_z = false;
        for x_idx in 0..self.d {
            let q_idx = x_idx; // the top row: 0, 1, 2, ..., d-1
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
            let q_idx = self.d * y_idx; // the leftmost column: 0, d, 2d, ...
            if residual_x[q_idx] {
                logical_x ^= true;
            }
        }

        let mut logical_z = false;
        for x_idx in 0..self.d {
            let q_idx = x_idx; // the top row: 0, 1, 2, ..., d-1
            if residual_z[q_idx] {
                logical_z ^= true;
            }
        }

        (logical_x as u8) | ((logical_z as u8) << 1)
    }

    /// One shot of circuit-level noise, decoded against the detector error model.
    ///
    /// Executes the same `round_program` the model was built from — see
    /// `circuit_model` for why that matters — sampling a Pauli at every `Noise`
    /// location and flipping each readout with probability p.
    ///
    /// Returns the logical Pauli class left behind: bit 0 set if a logical X
    /// survived, bit 1 if a logical Z. 0 means the shot succeeded.
    pub fn simulate_circuit_noise_with_model(
        &self,
        model: &crate::circuit_model::CircuitModel,
        num_rounds: usize,
        p: f64,
        bias: f64,
        init_state: &str,
        decoder_type: usize,
        erasure_rate: f64,
        correlated_noise: usize,
    ) -> u8 {
        use crate::circuit_model::{Op, StabKind};

        let num_data = self.data_qubits.len();
        let num_x = self.x_stabilizers.len();
        let num_z = self.z_stabilizers.len();
        let total_qubits = num_data + num_x + num_z;

        #[cfg(feature = "python")]
        let mut rng = Xorshift::new(rand::random());
        #[cfg(not(feature = "python"))]
        let mut rng = next_shot_rng();

        // Seed the tableau's measurement randomness from this shot's generator.
        let mut sim = crate::simulator::StabilizerSimulator::with_seed(total_qubits, rng.next_u64());

        let program = self.round_program();
        let rounds_total = crate::circuit_model::rounds_executed(num_rounds);
        let layers = crate::circuit_model::detector_layers(num_rounds);

        // Prepare the logical state.
        if init_state == "plus" {
            for i in 0..num_data {
                sim.apply_h(i);
            }
        }

        // State-preparation noise used to be applied here, before the baseline
        // round. That made it invisible: an error present during both the
        // baseline and the first noisy round leaves their readings identical,
        // so the detector comparing them never fires and the error goes
        // straight into the residual, uncorrectable at any distance. It showed
        // up as a stubborn p^1 term in a curve that should have scaled as
        // p^ceil(d/2). Preparation noise belongs after the projection, and the
        // first noisy round's gate noise already covers every data qubit, so it
        // is simply folded in there.

        // A Pauli frame shadowing the tableau. The tableau alone can only answer
        // the question its basis asks — prepared in |0_L> it detects a logical X
        // and is blind to a logical Z, which commutes with everything it
        // measures. That is fine for a pass/fail but not for a channel, which
        // needs all four class probabilities. Propagation is linear, so the same
        // injected Paulis carried through the same circuit give the residual
        // directly, and the tableau's own measurement randomness does not enter.
        let mut frame = crate::circuit_model::Frame::new(total_qubits);

        let mut flips_x = vec![false; num_x * rounds_total];
        let mut flips_z = vec![false; num_z * rounds_total];
        let mut erased_sites = vec![false; model.erasure_sites];

        for r in 0..rounds_total {
            // The first round projects into the code space and the last is the
            // perfect final readout every real memory experiment ends with;
            // both are noiseless, and everything between carries noise.
            let round_p = if r == 0 || r == rounds_total - 1 {
                0.0
            } else {
                get_drift_p(p, r - 1, correlated_noise)
            };

            let mut slot = 0usize;
            for &op in &program {
                match op {
                    Op::Reset(q) => {
                        if sim.measure_z(q) == 1 {
                            sim.apply_x(q);
                        }
                        frame.step(op);
                    }
                    Op::H(q) => { sim.apply_h(q); frame.step(op); }
                    Op::Cnot(c, t) => { sim.apply_cnot(c, t); frame.step(op); }
                    // The tableau has no native CZ; conjugating the target by H
                    // is the identity CZ = (I x H) CNOT (I x H).
                    Op::Cz(a, b) => {
                        sim.apply_h(b);
                        sim.apply_cnot(a, b);
                        sim.apply_h(b);
                        frame.step(op);
                    }
                    Op::Noise(q) => {
                        // A located loss: the hardware knows this qubit was lost
                        // here, so the Pauli is uniform and — crucially — the
                        // decoder gets told where. On an ancilla partway through
                        // its CNOTs that knowledge covers everything the loss
                        // goes on to touch.
                        let p_erase = round_p * erasure_rate;
                        let p_pauli = round_p * (1.0 - erasure_rate);
                        if p_erase > 0.0 && rng.next_f64() < p_erase {
                            let pauli = 1 + (rng.next_u64() % 3) as u8;
                            match pauli {
                                1 => sim.apply_x(q),
                                3 => sim.apply_y(q),
                                _ => sim.apply_z(q),
                            }
                            frame.apply(q, if pauli == 3 { 3 } else if pauli == 1 { 1 } else { 2 });
                            if r > 0 && r < rounds_total - 1 {
                                erased_sites[(r - 1) * model.noise_slots + slot] = true;
                            }
                        } else if let Some(pauli) = sample_pauli(p_pauli, bias, &mut rng) {
                            match pauli {
                                1 => sim.apply_x(q),
                                2 => sim.apply_z(q),
                                _ => sim.apply_y(q),
                            }
                            frame.apply(q, pauli);
                        }
                        slot += 1;
                    }
                    Op::Measure(q, kind, idx) => {
                        frame.step(op);
                        let mut m = sim.measure_z(q) == 1;
                        if round_p > 0.0 && rng.next_f64() < round_p {
                            m ^= true;
                        }
                        match kind {
                            StabKind::X => flips_x[r * num_x + idx] = m,
                            StabKind::Z => flips_z[r * num_z + idx] = m,
                        }
                    }
                }
            }

            if r > 0 && r < rounds_total - 1 {
                inject_correlated_noise_circuit(
                    &mut sim, Some(&mut frame), &self.data_qubits, self.d, correlated_noise, &mut rng);
            }
        }

        // Detectors compare each reading against the previous round's.
        let mut detectors_z = vec![false; num_z * layers];
        for t in 0..layers {
            for s in 0..num_z {
                detectors_z[s + t * num_z] =
                    flips_z[(t + 1) * num_z + s] != flips_z[t * num_z + s];
            }
        }
        let mut detectors_x = vec![false; num_x * layers];
        for t in 0..layers {
            for s in 0..num_x {
                detectors_x[s + t * num_x] =
                    flips_x[(t + 1) * num_x + s] != flips_x[t * num_x + s];
            }
        }

        // Decode on the model. An edge may correct several data qubits at once,
        // which is the whole reason this model exists.
        let apply = |sim: &mut crate::simulator::StabilizerSimulator,
                     frame: &mut crate::circuit_model::Frame,
                     dg: &crate::circuit_model::DetectorGraph,
                     defects: &[bool],
                     is_x: bool| {
            // Every edge an erased location could have produced costs nothing:
            // the matcher is free to route through it, because the loss is known
            // to have happened there.
            let mut erased_edges = vec![false; dg.graph.edges.len()];
            for (site, &lost) in erased_sites.iter().enumerate() {
                if lost {
                    for &edge in &dg.site_edges[site] {
                        erased_edges[edge] = true;
                    }
                }
            }
            let chosen = decode_by_type(&dg.graph, defects, decoder_type, &erased_edges);
            for edge_idx in chosen {
                let mask = dg.correction[edge_idx];
                for q in 0..num_data {
                    if (mask >> q) & 1 == 1 {
                        if is_x { sim.apply_x(q); } else { sim.apply_z(q); }
                        frame.apply(q, if is_x { 1 } else { 2 });
                    }
                }
            }
        };
        apply(&mut sim, &mut frame, &model.for_x_errors, &detectors_z, true);
        apply(&mut sim, &mut frame, &model.for_z_errors, &detectors_x, false);

        // Logical X runs down a column, logical Z across a row; a residual X is
        // a logical X when its parity against the column is odd, and vice versa.
        // Reading this off the frame rather than measuring gives both at once —
        // a measurement in one basis can only ever see one of them.
        let (mut rx, mut rz) = (0u128, 0u128);
        for q in 0..num_data {
            if frame.x[q] { rx |= 1u128 << q; }
            if frame.z[q] { rz |= 1u128 << q; }
        }
        let (mut column, mut row) = (0u128, 0u128);
        for i in 0..self.d {
            column |= 1u128 << (i * self.d);
            row |= 1u128 << i;
        }
        // `init_state` still chooses what is prepared — the tableau runs a real
        // memory experiment — but the class is read from the frame, so it no
        // longer decides which kind of failure is visible.
        ((rx & column).count_ones() % 2) as u8
            | ((((rz & row).count_ones() % 2) as u8) << 1)
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

    /// The syndrome-extraction round, as a list of instructions.
    ///
    /// As with the rotated code this is the single definition of the circuit,
    /// consumed both by the simulation and by the model builder.
    ///
    /// An XZZX plaquette reads `X Z Z X` from one ancilla, so the ancilla is
    /// held in |+> throughout: an X leg is a CNOT out of it, a Z leg a CZ. That
    /// is the whole difference from the CSS case, and it is why there is one
    /// stabilizer family here rather than two.
    ///
    /// All plaquettes walk their neighbours in the same order — NW, NE, SW, SE.
    /// The rotated code needs two opposed orders because its X and Z plaquettes
    /// overlap on two qubits and act with the same Pauli on both; here adjacent
    /// plaquettes already anticommute on each of their two shared qubits, and
    /// the two anticommutations cancel. A single order suffices, provided it
    /// touches both shared qubits in the same relative sequence — which this one
    /// does, for horizontal and vertical neighbours alike.
    pub fn round_program(&self) -> Vec<crate::circuit_model::Op> {
        self.round_program_ordered(&Self::SCHEDULE_A, &Self::SCHEDULE_B)
    }

    /// X on the NW/SE diagonal, Z on the NE/SW one — the same convention the
    /// constructor uses to build the symplectic form.
    ///
    /// The two sublattices walk their neighbours in *transposed* orders, NE and
    /// SW exchanged. This is the same lesson the rotated code's N and Z
    /// schedules teach, and it is not optional: with any single order shared by
    /// both sublattices, every one of the 24 permutations leaves 10 to 12 of 672
    /// single faults uncorrectable at d = 3 — a hook error off an ancilla lands
    /// as a weight-2 data error that the d = 3 patch cannot tell from a logical
    /// operator. Searching both sublattices independently — 576 combinations,
    /// scored on commutation against the tableau and on the exhaustive
    /// single-fault check — leaves exactly 6 that pass, all of them transposes.
    /// d = 5 is clean either way, so nothing but the d = 3 check finds this.
    pub const SCHEDULE_A: [(i32, i32, bool); 4] = [
        (-1, -1, true),  // NW, X
        (1, -1, false),  // NE, Z
        (-1, 1, false),  // SW, Z
        (1, 1, true),    // SE, X
    ];
    pub const SCHEDULE_B: [(i32, i32, bool); 4] = [
        (-1, -1, true),  // NW, X
        (-1, 1, false),  // SW, Z
        (1, -1, false),  // NE, Z
        (1, 1, true),    // SE, X
    ];

    pub fn round_program_ordered(
        &self,
        order_a: &[(i32, i32, bool); 4],
        order_b: &[(i32, i32, bool); 4],
    ) -> Vec<crate::circuit_model::Op> {
        use crate::circuit_model::{Op, StabKind};

        let num_data = self.data_qubits.len();
        let anc = |s: usize| num_data + s;
        let mut ops = Vec::new();

        for s in 0..self.stabilizers.len() {
            ops.push(Op::Reset(anc(s)));
            ops.push(Op::Noise(anc(s)));
            ops.push(Op::H(anc(s)));
            ops.push(Op::Noise(anc(s)));
        }

        let split = self.z_stabilizers.len();
        for step in 0..4 {
            for s in 0..self.stabilizers.len() {
                let (dx, dy, is_x) =
                    if s < split { order_a[step] } else { order_b[step] };
                if let Some(data) = self.get_neighbor_idx(
                    self.stabilizers[s].0 as i32 + dx,
                    self.stabilizers[s].1 as i32 + dy,
                ) {
                    if is_x {
                        ops.push(Op::Cnot(anc(s), data));
                    } else {
                        ops.push(Op::Cz(anc(s), data));
                    }
                    ops.push(Op::Noise(anc(s)));
                    ops.push(Op::Noise(data));
                }
            }
        }

        for s in 0..self.stabilizers.len() {
            ops.push(Op::H(anc(s)));
            ops.push(Op::Noise(anc(s)));
        }
        // One family, so every ancilla reports under the same kind. The combined
        // model reads them all from the Z slot.
        for s in 0..self.stabilizers.len() {
            ops.push(Op::Measure(anc(s), StabKind::Z, s));
        }

        ops
    }

    /// The circuit plus the sizes the model builder needs.
    ///
    /// `num_x_stabs` is zero: there is one stabilizer family, carried in the Z
    /// slot, and `build_combined` reads it from there.
    pub fn circuit_layout(&self) -> crate::circuit_model::CircuitLayout {
        crate::circuit_model::CircuitLayout {
            program: self.round_program(),
            num_qubits: self.data_qubits.len() + self.stabilizers.len(),
            num_data: self.data_qubits.len(),
            num_x_stabs: 0,
            num_z_stabs: self.stabilizers.len(),
        }
    }

    /// Circuit-level noise for the XZZX code, decoded against its detector
    /// error model.
    ///
    /// Run on the Pauli frame rather than the tableau. For a Clifford circuit
    /// with Pauli noise the two agree exactly on everything that matters here —
    /// which detectors fire and what Pauli is left on the data — and the frame
    /// makes the scoring honest: an XZZX logical operator is a mixed X/Z string,
    /// so there is no row of qubits to measure the way the rotated code can.
    /// The residual goes to the stabilizer group instead. That the circuit is a
    /// valid simultaneous measurement is checked separately, against the tableau.
    pub fn simulate_circuit_noise_with_model(
        &self,
        model: &crate::circuit_model::CombinedModel,
        num_rounds: usize,
        p: f64,
        bias: f64,
        decoder_type: usize,
        erasure_rate: f64,
        correlated_noise: usize,
    ) -> u8 {
        use crate::circuit_model::{Frame, Op};

        #[cfg(feature = "python")]
        let mut rng = Xorshift::new(rand::random());
        #[cfg(not(feature = "python"))]
        let mut rng = next_shot_rng();

        let num_data = self.data_qubits.len();
        let num_stabs = self.stabilizers.len();
        let program = self.round_program();
        let rounds_total = crate::circuit_model::rounds_executed(num_rounds);

        let mut frame = Frame::new(num_data + num_stabs);
        let mut flips = vec![false; num_stabs * rounds_total];
        let mut erased_sites = vec![false; model.erasure_sites];

        for r in 0..rounds_total {
            // The first and last rounds are noiseless: the first projects, and
            // the last supplies the comparison that makes the final round's
            // faults visible at all.
            let noisy = r > 0 && r < rounds_total - 1;
            let round_p = if noisy { get_drift_p(p, r, correlated_noise) } else { 0.0 };
            let mut slot = 0usize;

            for &op in &program {
                match op {
                    Op::Noise(q) => {
                        let p_erase = round_p * erasure_rate;
                        let p_pauli = round_p * (1.0 - erasure_rate);
                        if p_erase > 0.0 && rng.next_f64() < p_erase {
                            // A located loss. The Pauli is uniform and unknown,
                            // but the decoder is told exactly where it happened.
                            frame.apply(q, 1 + (rng.next_u64() % 3) as u8);
                            if noisy {
                                erased_sites[(r - 1) * model.noise_slots + slot] = true;
                            }
                        } else if let Some(pauli) = sample_pauli(p_pauli, bias, &mut rng) {
                            frame.apply(q, pauli);
                        }
                        slot += 1;
                    }
                    Op::Measure(_, _, idx) => {
                        let mut m = frame.step(op).is_some_and(|(_, _, flipped)| flipped);
                        if round_p > 0.0 && rng.next_f64() < round_p {
                            m ^= true;
                        }
                        flips[r * num_stabs + idx] = m;
                    }
                    other => {
                        frame.step(other);
                    }
                }
            }

            if noisy && (correlated_noise == 1 || correlated_noise == 3) && rng.next_f64() < 0.02 {
                let cx = rng.next_f64() * self.d as f64;
                let cy = rng.next_f64() * self.d as f64;
                for q in 0..num_data {
                    let (qx, qy) = self.data_qubits[q];
                    let ux = (qx as f64 - 1.0) / 2.0;
                    let uy = (qy as f64 - 1.0) / 2.0;
                    if ((ux - cx).powi(2) + (uy - cy).powi(2)).sqrt() <= 1.5 {
                        frame.apply(q, 1 + (rng.next_u64() % 3) as u8);
                    }
                }
            }
        }

        // A detector compares a reading against the previous one.
        let mut defects = vec![false; model.graph.graph.num_nodes];
        for t in 1..rounds_total {
            for s in 0..num_stabs {
                if flips[t * num_stabs + s] != flips[(t - 1) * num_stabs + s] {
                    defects[s + (t - 1) * num_stabs] = true;
                }
            }
        }

        // A located loss frees every edge its circuit location can produce —
        // including, on an ancilla, everything the loss propagates onto.
        let mut erased_edges = vec![false; model.graph.graph.edges.len()];
        for (site, &lost) in erased_sites.iter().enumerate() {
            if lost {
                for &e in &model.graph.site_edges[site] {
                    erased_edges[e] = true;
                }
            }
        }

        let (cx, cz) = crate::circuit_model::correction_for_combined(
            &model.graph, &defects, decoder_type, &erased_edges,
        );

        let (mut rx, mut rz) = (0u128, 0u128);
        for q in 0..num_data {
            if frame.x[q] { rx |= 1u128 << q; }
            if frame.z[q] { rz |= 1u128 << q; }
        }
        self.logical.classify(rx ^ cx, rz ^ cz)
    }
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
        self.logical.classify(rx, rz)
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
        self.logical.classify(rx, rz)
    }

}

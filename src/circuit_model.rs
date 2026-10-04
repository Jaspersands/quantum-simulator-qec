//! Detector error model for circuit-level noise.
//!
//! WHY THIS EXISTS
//! ---------------
//! Phenomenological noise puts errors on data qubits and lies in the readout,
//! and every elementary fault moves at most two detectors — so the spacetime
//! graph the matcher walks is a faithful map of what can go wrong. Circuit-level
//! noise is not like that. A fault on an ancilla partway through its four CNOTs
//! propagates onto every data qubit it has yet to touch, so one fault becomes a
//! *correlated multi-qubit* data error. The phenomenological graph has no edge
//! for that, so the matcher explains it with two unrelated edges and can walk
//! the correction straight into a logical operator. Measured before this module
//! existed: about 3% of single faults produced a logical error, and the code got
//! worse with distance rather than better.
//!
//! The fix is to stop guessing at the graph and derive it. For every elementary
//! fault the circuit admits, propagate it and record which detectors it fires
//! and what data error it leaves behind. That mapping *is* the decoding graph.
//!
//! TWO THINGS MAKE THIS TRACTABLE
//! ------------------------------
//! **The circuit is defined once.** `round_program` returns the round as a list
//! of `Op`, and both the stabilizer simulation and the fault propagation below
//! consume that same list. A detector error model that describes a circuit
//! subtly different from the one being run is worse than no model at all, and
//! keeping two hand-written copies in step is exactly the kind of thing that
//! rots. There is one copy.
//!
//! **Decomposition comes for free.** The usual hard part of building one of
//! these is that a fault firing three or more detectors has to be split into
//! graph-like pieces. Here the decoder already matches twice — Z-stabilizer
//! detectors catch X errors, X-stabilizer detectors catch Z errors — so a fault
//! is split by Pauli type: its X part goes to one graph, its Z part to the
//! other, and each is graph-like on its own.
//!
//! Faults propagate as a Pauli frame rather than through the full tableau: H
//! swaps x and z, CNOT sends x_target ^= x_control and z_control ^= z_target,
//! and a Z-basis measurement flips exactly when the frame carries x. That is
//! linear in the gate count instead of quadratic in the qubit count.

use std::collections::BTreeMap;

use crate::decoder::{Edge, SyndromeGraph};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StabKind {
    X,
    Z,
}

/// One instruction of a syndrome-extraction round.
#[derive(Clone, Copy, Debug)]
pub enum Op {
    /// Prepare an ancilla in |0>.
    Reset(usize),
    H(usize),
    Cnot(usize, usize),
    /// Controlled-Z. Symmetric, and the natural way to read a Z leg with an
    /// ancilla held in |+> — which is what an XZZX plaquette needs, since it
    /// reads X on one diagonal and Z on the other from a single ancilla.
    Cz(usize, usize),
    /// Measure an ancilla in Z and reset it. Records one outcome.
    Measure(usize, StabKind, usize),
    /// A location where the hardware may apply a single-qubit Pauli.
    Noise(usize),
}

/// A single-qubit Pauli, as a bitmask: 1 = X, 2 = Z, 3 = Y.
pub type Pauli = u8;

/// Where a fault can occur, and what it does there.
#[derive(Clone, Copy, Debug)]
pub enum Fault {
    /// A Pauli at a `Noise` location: (op index within the round, Pauli).
    Gate(usize, Pauli),
    /// A classical flip of a recorded outcome: (op index within the round).
    Readout(usize),
}

/// A decoding graph derived from the circuit, with the correction each edge
/// implies. Unlike the phenomenological graph an edge may touch several data
/// qubits — that is precisely the hook error the old graph could not express.
pub struct DetectorGraph {
    pub graph: SyndromeGraph,
    /// Data qubits each edge corrects, as a bitmask.
    pub correction: Vec<u128>,
    /// For each erasure site, the edges that become free when it is erased.
    ///
    /// An erasure is a *located* loss: the hardware knows the qubit was lost, so
    /// the decoder knows a uniformly random Pauli struck that exact circuit
    /// location. Every edge that location can produce therefore costs nothing,
    /// and the matcher is told so by weighting them zero. On an ancilla midway
    /// through its CNOTs that is several edges at once — the same propagation
    /// that makes a hook error — which is why this has to come from the model
    /// rather than from a qubit index.
    pub site_edges: Vec<Vec<usize>>,
}

/// The model for one (code, round count).
pub struct CircuitModel {
    /// Detectors are Z-stabilizer measurements; edges correct X errors.
    pub for_x_errors: DetectorGraph,
    /// Detectors are X-stabilizer measurements; edges correct Z errors.
    pub for_z_errors: DetectorGraph,
    pub num_rounds: usize,
    /// Noise locations per round, so a caller can index erasure sites as
    /// `(round - 1) * noise_slots + slot`.
    pub noise_slots: usize,
    /// Total erasure sites across the noisy rounds.
    pub erasure_sites: usize,
}

/// Pauli frame over data qubits and ancillas.
///
/// For a Clifford circuit under Pauli noise this is a complete description: the
/// noiseless circuit fixes what every measurement would read, and the frame says
/// which of those readings are flipped. That is all a detector needs, and it is
/// what lets a non-CSS code be simulated without tracking a tableau — provided
/// the circuit really is a valid simultaneous measurement, which is a separate
/// property and is tested against the tableau instead.
pub(crate) struct Frame {
    pub(crate) x: Vec<bool>,
    pub(crate) z: Vec<bool>,
}

impl Frame {
    pub(crate) fn new(n: usize) -> Self {
        Frame { x: vec![false; n], z: vec![false; n] }
    }

    pub(crate) fn apply(&mut self, q: usize, pauli: Pauli) {
        if pauli & 1 != 0 { self.x[q] ^= true; }
        if pauli & 2 != 0 { self.z[q] ^= true; }
    }

    /// Walk one op. Returns Some(outcome flipped) for a measurement.
    pub(crate) fn step(&mut self, op: Op) -> Option<(StabKind, usize, bool)> {
        match op {
            Op::Reset(q) => {
                self.x[q] = false;
                self.z[q] = false;
                None
            }
            Op::H(q) => {
                std::mem::swap(&mut self.x[q], &mut self.z[q]);
                None
            }
            Op::Cnot(c, t) => {
                // X propagates control -> target, Z propagates target -> control.
                if self.x[c] { self.x[t] ^= true; }
                if self.z[t] { self.z[c] ^= true; }
                None
            }
            Op::Cz(a, b) => {
                // CZ is symmetric and turns an X on either side into a Z on the
                // other; Z commutes with it and does not move.
                if self.x[a] { self.z[b] ^= true; }
                if self.x[b] { self.z[a] ^= true; }
                None
            }
            Op::Measure(q, kind, idx) => {
                // A Z-basis measurement is flipped by an X in the frame; a Z
                // there commutes with it and changes nothing.
                let flipped = self.x[q];
                self.x[q] = false;
                self.z[q] = false;
                Some((kind, idx, flipped))
            }
            Op::Noise(_) => None,
        }
    }
}

/// What one fault does: which measurement outcomes it flips, and the data-qubit
/// Pauli it leaves behind at the end.
pub(crate) struct FaultEffect {
    /// Indexed `round * num_stabs + stabilizer`, over the baseline round plus
    /// every noisy round.
    pub(crate) flips_x_stab: Vec<bool>,
    pub(crate) flips_z_stab: Vec<bool>,
    pub(crate) residual_x: u128,
    pub(crate) residual_z: u128,
}

pub struct CircuitLayout {
    pub program: Vec<Op>,
    pub num_qubits: usize,
    pub num_data: usize,
    pub num_x_stabs: usize,
    pub num_z_stabs: usize,
}

impl CircuitLayout {
    /// Propagate a single fault injected in `round` and report its effect.
    ///
    /// `rounds_total` counts the baseline round plus the noisy ones, so a fault
    /// injected before the baseline correctly cancels itself out of the first
    /// detector — which is what makes initialisation noise harmless when it is.
    pub(crate) fn propagate(&self, fault: Fault, round: usize, rounds_total: usize) -> FaultEffect {
        let mut frame = Frame::new(self.num_qubits);
        let mut flips_x = vec![false; self.num_x_stabs * rounds_total];
        let mut flips_z = vec![false; self.num_z_stabs * rounds_total];

        for r in 0..rounds_total {
            for (i, &op) in self.program.iter().enumerate() {
                // Inject before the op at a gate location, so the Pauli lands on
                // the state the gate is about to act on.
                if r == round {
                    if let Fault::Gate(at, pauli) = fault {
                        if at == i {
                            if let Op::Noise(q) = op {
                                frame.apply(q, pauli);
                            }
                        }
                    }
                }

                if let Some((kind, idx, mut flipped)) = frame.step(op) {
                    if r == round {
                        if let Fault::Readout(at) = fault {
                            if at == i {
                                flipped ^= true;
                            }
                        }
                    }
                    match kind {
                        StabKind::X => flips_x[r * self.num_x_stabs + idx] = flipped,
                        StabKind::Z => flips_z[r * self.num_z_stabs + idx] = flipped,
                    }
                }
            }
        }

        let mut residual_x = 0u128;
        let mut residual_z = 0u128;
        for q in 0..self.num_data {
            if frame.x[q] { residual_x |= 1u128 << q; }
            if frame.z[q] { residual_z |= 1u128 << q; }
        }

        FaultEffect { flips_x_stab: flips_x, flips_z_stab: flips_z, residual_x, residual_z }
    }

    /// Every elementary fault one round admits, paired with its erasure slot.
    ///
    /// A gate fault belongs to the noise location it sits at — that is the thing
    /// an erasure can be *located* to. Readout flips have no slot; losing a
    /// qubit and misreading a measurement are different failures.
    pub(crate) fn fault_locations(&self) -> Vec<(Fault, Option<usize>)> {
        let mut out = Vec::new();
        let mut slot = 0usize;
        for (i, op) in self.program.iter().enumerate() {
            match op {
                Op::Noise(_) => {
                    for pauli in 1..=3u8 {
                        out.push((Fault::Gate(i, pauli), Some(slot)));
                    }
                    slot += 1;
                }
                Op::Measure(..) => out.push((Fault::Readout(i), None)),
                _ => {}
            }
        }
        out
    }

    /// How many noise locations one round has.
    pub fn noise_slots(&self) -> usize {
        self.program.iter().filter(|op| matches!(op, Op::Noise(_))).count()
    }
}

/// Number of rounds actually executed: a noiseless projection round, the noisy
/// rounds, and a noiseless final round.
///
/// The final round is not decoration. Detectors compare a reading against the
/// previous one, so without a round after the last noisy one, anything that
/// goes wrong in that last round is invisible — there is nothing to compare it
/// against. A real memory experiment ends with a transversal data readout that
/// supplies exactly this comparison. Leaving it out made roughly one single
/// fault in twenty uncorrectable no matter how large the patch.
pub fn rounds_executed(num_rounds: usize) -> usize {
    num_rounds + 2
}

/// Number of detector layers: one per comparison between consecutive rounds.
pub fn detector_layers(num_rounds: usize) -> usize {
    num_rounds + 1
}

/// Detector index from a measurement flip pattern.
///
/// A detector compares a stabilizer's reading against its previous one, so the
/// detector for round t fires when the flips at t and t-1 differ. Round 0 of the
/// noisy sequence is compared against the baseline round.
fn detectors(flips: &[bool], num_stabs: usize, rounds_total: usize) -> Vec<usize> {
    let mut fired = Vec::new();
    for t in 1..rounds_total {
        for s in 0..num_stabs {
            let here = flips[t * num_stabs + s];
            let before = flips[(t - 1) * num_stabs + s];
            if here != before {
                // Detector layers are indexed from the first noisy round.
                fired.push(s + (t - 1) * num_stabs);
            }
        }
    }
    fired
}

/// Collect the edges of one graph, keeping track of which erasure site each came
/// from so a located loss can name the edges it makes free.
fn assemble(
    candidates: Vec<(Vec<usize>, u128, usize)>,
    num_nodes: usize,
    num_sites: usize,
) -> DetectorGraph {
    // Exactly one edge per node pair, carrying the most likely correction for
    // it. A matching decoder can only ever decide "connect u and v" — it has no
    // way to choose between two mechanisms that connect the same pair — so
    // emitting both as parallel edges does not give it a choice, it just makes
    // the choice arbitrary: whichever edge the shortest-path search happens to
    // relax first supplies the correction. Every elementary fault in this model
    // carries the same probability, so the most likely correction for a pair is
    // simply the one the most mechanisms produce. Ties go to the smaller mask,
    // which keeps the model reproducible rather than dependent on enumeration
    // order.
    let mut tally: BTreeMap<(usize, usize), BTreeMap<u128, usize>> = BTreeMap::new();
    let endpoints = |nodes: &Vec<usize>| -> Option<(usize, usize)> {
        match nodes.len() {
            1 => Some((nodes[0], num_nodes)), // to the boundary
            2 => Some((nodes[0], nodes[1])),
            _ => None,
        }
    };
    for (nodes, corr, _) in &candidates {
        if let Some(pair) = endpoints(nodes) {
            *tally.entry(pair).or_default().entry(*corr).or_insert(0) += 1;
        }
    }

    let mut edges = Vec::new();
    let mut edge_to_qubit = Vec::new();
    let mut correction = Vec::new();
    let mut edge_of_pair: BTreeMap<(usize, usize), usize> = BTreeMap::new();

    for (&(u, v), corrs) in &tally {
        let (&best, _) = corrs
            .iter()
            .max_by_key(|(mask, count)| (**count, std::cmp::Reverse(**mask)))
            .expect("a tallied pair has at least one correction");
        // The matcher's own bookkeeping wants a representative qubit; the real
        // correction is the mask alongside.
        let representative = if best == 0 { None } else { Some(best.trailing_zeros() as usize) };
        edge_of_pair.insert((u, v), edges.len());
        edges.push(Edge { u, v, id: edges.len() });
        edge_to_qubit.push(representative);
        correction.push(best);
    }

    let mut site_edges = vec![Vec::new(); num_sites];
    for (nodes, _, site) in &candidates {
        if let Some(pair) = endpoints(nodes) {
            if let Some(&id) = edge_of_pair.get(&pair) {
                if !site_edges[*site].contains(&id) {
                    site_edges[*site].push(id);
                }
            }
        }
    }

    DetectorGraph {
        graph: SyndromeGraph { num_nodes, edges, edge_to_qubit },
        correction,
        site_edges,
    }
}

/// Exhaustively check that every single fault the circuit admits is corrected.
///
/// A distance-d code must survive any one fault. This is deterministic and
/// complete — no sampling — so it either passes or names the mechanism that
/// breaks it. It is the test that says whether the model is right, and it runs
/// entirely on the Pauli frame: propagate the fault, decode its detectors
/// against the model, apply the correction, and see what is left.
///
/// `logical_x_support` and `logical_z_support` are qubit masks whose parity
/// against the residual detects each logical operator.
pub fn single_fault_failures(
    layout: &CircuitLayout,
    model: &CircuitModel,
    num_rounds: usize,
    decoder_type: usize,
    logical_x_support: u128,
    logical_z_support: u128,
) -> (usize, usize) {
    let rounds_total = rounds_executed(num_rounds);
    let mut tested = 0usize;
    let mut failures = 0usize;

    for round in 1..rounds_total - 1 {
        for (fault, _) in layout.fault_locations() {
            let effect = layout.propagate(fault, round, rounds_total);
            tested += 1;

            let dz = detectors(&effect.flips_z_stab, layout.num_z_stabs, rounds_total);
            let dx = detectors(&effect.flips_x_stab, layout.num_x_stabs, rounds_total);

            let mut residual_x = effect.residual_x;
            let mut residual_z = effect.residual_z;

            // X errors, decoded on the Z-stabilizer detectors.
            let mut defects = vec![false; model.for_x_errors.graph.num_nodes];
            for &n in &dz { defects[n] = true; }
            let none_x = vec![false; model.for_x_errors.graph.edges.len()];
            residual_x ^= correction_for(&model.for_x_errors, &defects, decoder_type, &none_x);

            // Z errors, decoded on the X-stabilizer detectors.
            let mut defects = vec![false; model.for_z_errors.graph.num_nodes];
            for &n in &dx { defects[n] = true; }
            let none_z = vec![false; model.for_z_errors.graph.edges.len()];
            residual_z ^= correction_for(&model.for_z_errors, &defects, decoder_type, &none_z);

            let logical_x = (residual_x & logical_z_support).count_ones() % 2 == 1;
            let logical_z = (residual_z & logical_x_support).count_ones() % 2 == 1;
            if logical_x || logical_z {
                failures += 1;
            }
        }
    }
    (tested, failures)
}

/// Decode one graph and collect the data-qubit correction it implies.
pub fn correction_for(
    dg: &DetectorGraph,
    defects: &[bool],
    decoder_type: usize,
    erased: &[bool],
) -> u128 {
    let chosen = match decoder_type {
        1 => crate::decoder::decode_greedy(&dg.graph, defects, erased),
        2 => crate::decoder::decode_mwpm(&dg.graph, defects, erased),
        _ => crate::decoder::decode_union_find(&dg.graph, defects, erased),
    };
    let mut mask = 0u128;
    for edge_idx in chosen {
        mask ^= dg.correction[edge_idx];
    }
    mask
}

/// How many faults fall into each detector-count bucket, per graph.
/// Anything landing above two is a fault the graph cannot express.
pub struct ModelStats {
    pub x_buckets: [usize; 5],
    pub z_buckets: [usize; 5],
    pub x_edges: usize,
    pub z_edges: usize,
}

/// Same enumeration as `build`, reporting where the faults land.
pub fn stats(layout: &CircuitLayout, num_rounds: usize) -> ModelStats {
    let rounds_total = rounds_executed(num_rounds);
    let mut st = ModelStats { x_buckets: [0; 5], z_buckets: [0; 5], x_edges: 0, z_edges: 0 };
    for round in 1..rounds_total - 1 {
        for (fault, _) in layout.fault_locations() {
            let effect = layout.propagate(fault, round, rounds_total);
            let dz = detectors(&effect.flips_z_stab, layout.num_z_stabs, rounds_total);
            let dx = detectors(&effect.flips_x_stab, layout.num_x_stabs, rounds_total);
            st.x_buckets[dz.len().min(4)] += 1;
            st.z_buckets[dx.len().min(4)] += 1;
        }
    }
    let model = build(layout, num_rounds);
    st.x_edges = model.for_x_errors.graph.edges.len();
    st.z_edges = model.for_z_errors.graph.edges.len();
    st
}

/// Build the model by enumerating every fault the circuit admits.
pub fn build(layout: &CircuitLayout, num_rounds: usize) -> CircuitModel {
    let rounds_total = rounds_executed(num_rounds);
    let layers = detector_layers(num_rounds);
    let num_x_nodes = layout.num_x_stabs * layers;
    let num_z_nodes = layout.num_z_stabs * layers;

    let noise_slots = layout.noise_slots();
    let erasure_sites = noise_slots * num_rounds;

    let mut for_x: Vec<(Vec<usize>, u128, usize)> = Vec::new();
    let mut for_z: Vec<(Vec<usize>, u128, usize)> = Vec::new();

    // The first and last rounds are noiseless, so faults are enumerated only in
    // the rounds that actually carry noise.
    for round in 1..rounds_total - 1 {
        for (fault, slot) in layout.fault_locations() {
            let effect = layout.propagate(fault, round, rounds_total);
            // Readout flips are not erasures, so they belong to no site; park
            // them on a sentinel that no erasure ever names.
            let site = slot
                .map(|k| (round - 1) * noise_slots + k)
                .unwrap_or(erasure_sites);

            // X errors show up in Z-stabilizer detectors.
            let dz = detectors(&effect.flips_z_stab, layout.num_z_stabs, rounds_total);
            if !dz.is_empty() && dz.len() <= 2 {
                for_x.push((dz, effect.residual_x, site));
            }

            // Z errors show up in X-stabilizer detectors.
            let dx = detectors(&effect.flips_x_stab, layout.num_x_stabs, rounds_total);
            if !dx.is_empty() && dx.len() <= 2 {
                for_z.push((dx, effect.residual_z, site));
            }
        }
    }

    CircuitModel {
        for_x_errors: assemble(for_x, num_z_nodes, erasure_sites + 1),
        for_z_errors: assemble(for_z, num_x_nodes, erasure_sites + 1),
        num_rounds,
        noise_slots,
        erasure_sites,
    }
}

/* -- Non-CSS codes: one graph, two error families ------------------------ */

/// A decoding graph for a code whose stabilizers are not separable into X and
/// Z checks.
///
/// The CSS model above splits into two independent graphs because a rotated
/// plaquette measures either X or Z, so X errors and Z errors light disjoint
/// detector sets. An XZZX plaquette measures `X Z Z X` — one ancilla, one
/// syndrome bit — so both families fire the *same* detectors and no such split
/// exists. Instead there is a single node set carrying both edge families, and
/// each edge names the X part and the Z part of the correction it implies.
pub struct CombinedGraph {
    pub graph: SyndromeGraph,
    pub correction_x: Vec<u128>,
    pub correction_z: Vec<u128>,
    /// For each erasure site, the edges that become free when it is erased.
    /// See `DetectorGraph::site_edges`.
    pub site_edges: Vec<Vec<usize>>,
}

pub struct CombinedModel {
    pub graph: CombinedGraph,
    pub num_rounds: usize,
    pub noise_slots: usize,
    pub erasure_sites: usize,
}

/// Collect edges for a combined graph, deduplicating on the full signature.
///
/// Unlike the CSS case the correction is a pair, so two faults that fire the
/// same detectors are still distinct mechanisms if they leave different Paulis.
fn assemble_combined(
    candidates: Vec<(Vec<usize>, u128, u128, usize)>,
    num_nodes: usize,
    num_sites: usize,
) -> CombinedGraph {
    // One edge per node pair, carrying the most likely correction. See
    // `assemble` for why parallel edges are not a choice the matcher can make.
    // It matters more here: on a single node set the competing mechanisms differ
    // by Pauli *type*, not merely by position, so an arbitrary pick applies an X
    // where a Z belonged.
    let endpoints = |nodes: &Vec<usize>| -> Option<(usize, usize)> {
        match nodes.len() {
            1 => Some((nodes[0], num_nodes)), // to the boundary
            2 => Some((nodes[0], nodes[1])),
            _ => None,
        }
    };
    let mut tally: BTreeMap<(usize, usize), BTreeMap<(u128, u128), usize>> = BTreeMap::new();
    for (nodes, cx, cz, _) in &candidates {
        if let Some(pair) = endpoints(nodes) {
            *tally.entry(pair).or_default().entry((*cx, *cz)).or_insert(0) += 1;
        }
    }

    let mut edges = Vec::new();
    let mut edge_to_qubit = Vec::new();
    let mut correction_x = Vec::new();
    let mut correction_z = Vec::new();
    let mut edge_of_pair: BTreeMap<(usize, usize), usize> = BTreeMap::new();

    for (&(u, v), corrs) in &tally {
        let (&(cx, cz), _) = corrs
            .iter()
            .max_by_key(|(masks, count)| (**count, std::cmp::Reverse(**masks)))
            .expect("a tallied pair has at least one correction");
        let representative = if cx != 0 {
            Some(cx.trailing_zeros() as usize)
        } else if cz != 0 {
            Some(cz.trailing_zeros() as usize)
        } else {
            None
        };
        edge_of_pair.insert((u, v), edges.len());
        edges.push(Edge { u, v, id: edges.len() });
        edge_to_qubit.push(representative);
        correction_x.push(cx);
        correction_z.push(cz);
    }

    let mut site_edges = vec![Vec::new(); num_sites];
    for (nodes, _, _, site) in &candidates {
        if let Some(pair) = endpoints(nodes) {
            if let Some(&id) = edge_of_pair.get(&pair) {
                if !site_edges[*site].contains(&id) {
                    site_edges[*site].push(id);
                }
            }
        }
    }

    CombinedGraph {
        graph: SyndromeGraph { num_nodes, edges, edge_to_qubit },
        correction_x,
        correction_z,
        site_edges,
    }
}

/// Build the detector error model for a non-CSS code.
///
/// Only X and Z faults are enumerated, never Y. Propagation is linear over the
/// Pauli frame, so a Y fault is exactly the composition of the X and Z faults at
/// the same location: it fires the XOR of their detectors and leaves the product
/// of their residuals. Enumerating it as well would add an edge firing up to
/// four detectors, which no graph can express — whereas letting the matcher pick
/// the two graphlike edges reconstructs it exactly. This is the decomposition
/// the CSS model got for free by matching twice.
pub fn build_combined(layout: &CircuitLayout, num_rounds: usize) -> CombinedModel {
    let rounds_total = rounds_executed(num_rounds);
    let layers = detector_layers(num_rounds);
    let num_nodes = layout.num_z_stabs * layers;

    let noise_slots = layout.noise_slots();
    let erasure_sites = noise_slots * num_rounds;

    let mut candidates: Vec<(Vec<usize>, u128, u128, usize)> = Vec::new();

    for round in 1..rounds_total - 1 {
        for (fault, slot) in layout.fault_locations() {
            if let Fault::Gate(_, 3) = fault {
                continue; // Y is X then Z; see above.
            }
            let effect = layout.propagate(fault, round, rounds_total);
            let site = slot
                .map(|k| (round - 1) * noise_slots + k)
                .unwrap_or(erasure_sites);

            let d = detectors(&effect.flips_z_stab, layout.num_z_stabs, rounds_total);
            if !d.is_empty() && d.len() <= 2 {
                candidates.push((d, effect.residual_x, effect.residual_z, site));
            }
        }
    }

    CombinedModel {
        graph: assemble_combined(candidates, num_nodes, erasure_sites + 1),
        num_rounds,
        noise_slots,
        erasure_sites,
    }
}

/// Decode a combined graph and collect the (X, Z) correction it implies.
pub fn correction_for_combined(
    cg: &CombinedGraph,
    defects: &[bool],
    decoder_type: usize,
    erased: &[bool],
) -> (u128, u128) {
    let chosen = match decoder_type {
        1 => crate::decoder::decode_greedy(&cg.graph, defects, erased),
        2 => crate::decoder::decode_mwpm(&cg.graph, defects, erased),
        _ => crate::decoder::decode_union_find(&cg.graph, defects, erased),
    };
    let (mut mx, mut mz) = (0u128, 0u128);
    for edge_idx in chosen {
        mx ^= cg.correction_x[edge_idx];
        mz ^= cg.correction_z[edge_idx];
    }
    (mx, mz)
}

/// How many faults land in each detector-count bucket for a combined graph.
/// Anything above two is a fault the graph cannot express.
pub fn stats_combined(layout: &CircuitLayout, num_rounds: usize) -> ([usize; 5], usize) {
    let rounds_total = rounds_executed(num_rounds);
    let mut buckets = [0usize; 5];
    for round in 1..rounds_total - 1 {
        for (fault, _) in layout.fault_locations() {
            if let Fault::Gate(_, 3) = fault {
                continue;
            }
            let effect = layout.propagate(fault, round, rounds_total);
            let d = detectors(&effect.flips_z_stab, layout.num_z_stabs, rounds_total);
            buckets[d.len().min(4)] += 1;
        }
    }
    let edges = build_combined(layout, num_rounds).graph.graph.edges.len();
    (buckets, edges)
}

/// Exhaustively check every single fault against a combined model.
///
/// `is_logical` asks the code whether a residual is a logical operator; for a
/// non-CSS code there is no straight row of qubits to measure, so the question
/// has to go to the stabilizer group itself.
pub fn single_fault_failures_combined(
    layout: &CircuitLayout,
    model: &CombinedModel,
    num_rounds: usize,
    decoder_type: usize,
    is_logical: &dyn Fn(u128, u128) -> bool,
) -> (usize, usize) {
    let rounds_total = rounds_executed(num_rounds);
    let mut tested = 0usize;
    let mut failures = 0usize;
    let none = vec![false; model.graph.graph.edges.len()];

    for round in 1..rounds_total - 1 {
        for (fault, _) in layout.fault_locations() {
            let effect = layout.propagate(fault, round, rounds_total);
            tested += 1;

            let d = detectors(&effect.flips_z_stab, layout.num_z_stabs, rounds_total);
            let mut defects = vec![false; model.graph.graph.num_nodes];
            for &n in &d { defects[n] = true; }

            let (cx, cz) = correction_for_combined(&model.graph, &defects, decoder_type, &none);
            if is_logical(effect.residual_x ^ cx, effect.residual_z ^ cz) {
                failures += 1;
            }
        }
    }
    (tested, failures)
}

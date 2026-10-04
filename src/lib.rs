#![doc = include_str!("../docs/crate.md")]
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

// The engine. Its modules are public so the Python bindings, the WebAssembly site, the tools and
// the fuzz targets can reach them, and hidden: they are the engine's internals, outside the
// stability promise. The crate's API is `api`, re-exported below.
#[doc(hidden)]
pub mod tableau;
#[doc(hidden)]
pub mod simulator;
#[doc(hidden)]
pub mod blossom;
#[doc(hidden)]
pub mod decoder;
#[doc(hidden)]
pub mod surface_code;
#[doc(hidden)]
pub mod circuit_model;
#[doc(hidden)]
pub mod css;
#[doc(hidden)]
pub mod circuit;
#[doc(hidden)]
pub mod gates;
#[doc(hidden)]
pub mod generated;
#[doc(hidden)]
pub mod dem;
#[doc(hidden)]
pub mod dem_build;
#[doc(hidden)]
pub mod dem_program;
#[doc(hidden)]
pub mod diagram;
#[doc(hidden)]
pub mod dem_decoder;
#[doc(hidden)]
pub mod sparse;
#[doc(hidden)]
pub mod m2d;
#[doc(hidden)]
pub mod window;
#[doc(hidden)]
pub mod bp;
#[doc(hidden)]
pub mod gf2;
#[doc(hidden)]
pub mod bb;
#[doc(hidden)]
pub mod bb_auto;
#[doc(hidden)]
pub mod bb_circuit;
#[doc(hidden)]
pub mod bb_gauge;
#[doc(hidden)]
pub mod osd;
#[doc(hidden)]
pub mod surgery;
#[doc(hidden)]
pub mod belief;
#[doc(hidden)]
pub mod stream;
#[doc(hidden)]
pub mod frame_sampler;
#[doc(hidden)]
pub mod batch_sampler;
#[doc(hidden)]
pub mod shots;
#[doc(hidden)]
pub mod memory;
#[doc(hidden)]
pub mod parallel;
#[doc(hidden)]
pub mod batch;
#[doc(hidden)]
pub mod fuzzing;
#[cfg(test)]
mod fixtures;
#[cfg(test)]
mod equivalence;
#[cfg(feature = "python")]
mod py_api;
#[cfg(feature = "python")]
mod py_objects;
#[cfg(not(feature = "python"))]
mod wasm_xc;
#[cfg(not(feature = "python"))]
mod wasm_hw;
#[cfg(not(feature = "python"))]
mod wasm_rt;
#[cfg(not(feature = "python"))]
mod wasm_bb;
#[cfg(not(feature = "python"))]
mod wasm_ls;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyclass(name = "RotatedSurfaceCode")]
struct PyRotatedSurfaceCode {
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

/// The extension, `stabilizer_qec._core`: the engine's bindings. The public
/// package (python/stabilizer_qec) is built on it.
#[cfg(feature = "python")]
#[pymodule]
#[pyo3(name = "_core")]
fn stabilizer_qec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRotatedSurfaceCode>()?;
    py_api::register(m)?;
    py_objects::register(m)?;
    Ok(())
}

mod wasm_session;

mod api;
pub use api::*;

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
                ((a.0 & b.1).count_ones() + (a.1 & b.0).count_ones()).is_multiple_of(2)
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

    /// Blossom must agree with brute force, exactly, wherever it answers at all.
    ///
    /// Exhaustive enumeration of perfect matchings is tractable to about ten
    /// vertices (945 of them at n = 10), so an exact oracle is available for
    /// precisely the sizes where a blossom implementation's bugs first appear.
    /// Instances where a ceiling trips return None and are counted separately —
    /// those cost the decoder nothing, since it keeps the matching it had, but a
    /// high rate of them would mean the implementation is not earning its place.
    #[test]
    fn blossom_agrees_with_brute_force() {
        fn brute(n: usize, cost: &[Vec<i64>]) -> i64 {
            fn go(used: &mut Vec<bool>, n: usize, cost: &[Vec<i64>], acc: i64, best: &mut i64) {
                let Some(u) = (0..n).find(|&i| !used[i]) else {
                    *best = (*best).min(acc);
                    return;
                };
                used[u] = true;
                for v in (u + 1)..n {
                    if !used[v] {
                        used[v] = true;
                        go(used, n, cost, acc + cost[u][v], best);
                        used[v] = false;
                    }
                }
                used[u] = false;
            }
            let mut used = vec![false; n];
            let mut best = i64::MAX;
            go(&mut used, n, cost, 0, &mut best);
            best
        }

        let mut rng: u64 = 0x243F6A8885A308D3;
        let mut next = move || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };

        let (mut agreed, mut declined) = (0usize, 0usize);
        let mut reasons: std::collections::BTreeMap<&'static str, usize> = Default::default();
        for n in [2usize, 4, 6, 8, 10, 12] {
            for trial in 0..250 {
                let mut cost = vec![vec![0i64; n]; n];
                if trial % 5 == 4 {
                    // Metric costs, which is what the decoder actually hands it:
                    // shortest-path distances obey the triangle inequality, and
                    // that changes which matchings are even competitive.
                    let pts: Vec<(i64, i64)> = (0..n)
                        .map(|_| ((next() % 40) as i64, (next() % 40) as i64))
                        .collect();
                    for i in 0..n {
                        for j in (i + 1)..n {
                            let w = (pts[i].0 - pts[j].0).abs() + (pts[i].1 - pts[j].1).abs();
                            cost[i][j] = w;
                            cost[j][i] = w;
                        }
                    }
                } else {
                    // Narrow ranges make ties, and ties are where blossoms form.
                    let span = [2i64, 5, 12, 60][trial % 4];
                    for i in 0..n {
                        for j in (i + 1)..n {
                            let w = (next() % (span as u64)) as i64 + 1;
                            cost[i][j] = w;
                            cost[j][i] = w;
                        }
                    }
                }
                let (maybe, why) = crate::blossom::min_weight_perfect_matching_diagnostic(n, &cost);
                let Some(mate) = maybe else {
                    declined += 1;
                    *reasons.entry(why).or_insert(0usize) += 1;
                    continue;
                };
                for i in 0..n {
                    assert_ne!(mate[i], i, "n={n} trial={trial}: {i} matched to itself");
                    assert_eq!(mate[mate[i]], i, "n={n} trial={trial}: asymmetric at {i}");
                }
                let got: i64 = (0..n).filter(|&i| i < mate[i]).map(|i| cost[i][mate[i]]).sum();
                let want = brute(n, &cost);
                assert_eq!(got, want, "n={n} trial={trial}: blossom {got}, brute {want}");
                agreed += 1;
            }
        }
        println!("blossom: {agreed} matched brute force, {declined} declined");
        for (why, count) in &reasons {
            println!("   declined: {count:>4}  {why}");
        }
        assert!(agreed > 0, "the matcher never answered");
    }

    #[test]
    #[ignore]
    fn blossom_speed() {
        use crate::blossom::min_weight_perfect_matching;
        let mut rng: u64 = 0x9E3779B97F4A7C15;
        let mut next = move || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
        for n in [20usize, 40, 60, 80, 120, 160] {
            let mut total = std::time::Duration::ZERO;
            let reps = 20;
            let mut declined = 0;
            for _ in 0..reps {
                let pts: Vec<(i64, i64)> = (0..n)
                    .map(|_| ((next() % 200) as i64, (next() % 200) as i64)).collect();
                let cost: Vec<Vec<i64>> = (0..n).map(|i| (0..n).map(|j|
                    (pts[i].0 - pts[j].0).abs() + (pts[i].1 - pts[j].1).abs()).collect()).collect();
                let t = std::time::Instant::now();
                if min_weight_perfect_matching(n, &cost).is_none() { declined += 1; }
                total += t.elapsed();
            }
            println!("  n={n:<4} {:>8.3} ms/solve   declined {declined}/{reps}",
                total.as_secs_f64() * 1000.0 / reps as f64);
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

//! The Python package's bindings (`stabilizer_qec`): circuits and error models, sampling,
//! plain, correlated and windowed matching, streams, BP and BP+OSD, belief-matching, the
//! bivariate bicycle codes and their logical operations, and lattice surgery.
//!
//! Shots cross the boundary as bytes in Stim's b8 layout (numpy's
//! `packbits(..., bitorder="little")`), and predictions come back as
//! little-endian u64 and f64 arrays, so no numpy crate is needed on this side.

// PyO3's #[pyfunction] expansion converts each PyResult's error into PyErr, which clippy
// flags on every binding as a useless conversion; it is the macro's, not this code's.
#![allow(clippy::useless_conversion)]

use std::time::Instant;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::circuit::{Basis, Circuit};
use crate::dem::Dem;
use crate::dem_decoder::{DemDecoder, Prediction};
use crate::frame_sampler::FrameSampler;
use crate::batch::{belief_shots, bposd_shots, match_shots, window_info, window_shots};
use crate::py_objects::check_rows;
use crate::memory::{generate, CodeKind, NoiseModel};
use crate::shots::{pack_row, read_b8, write_01};
use crate::surface_code::Xorshift;

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

/// `threads = 0` means every core; never more threads than units of work.
use crate::parallel::resolve_threads;

/// A memory experiment as a Stim circuit: a distance-`d` patch (`"rotated"`
/// or `"xzzx"`) held for `rounds` rounds, under `"sd6"` noise at `p` or the
/// biased `"current"` model with bias `eta`, in basis `"z"` or `"x"`.
#[pyfunction]
#[pyo3(signature = (code, d, rounds, noise, p, eta=0.5, basis="z"))]
fn generate_circuit(code: &str, d: usize, rounds: usize, noise: &str, p: f64, eta: f64, basis: &str) -> PyResult<String> {
    let kind = match code {
        "rotated" => CodeKind::Rotated,
        "xzzx" => CodeKind::Xzzx,
        _ => return Err(err(format!("unknown code '{code}'"))),
    };
    let noise = match noise {
        "current" => NoiseModel::Current { p, eta },
        "sd6" => NoiseModel::Sd6 { p },
        _ => return Err(err(format!("unknown noise model '{noise}'"))),
    };
    let basis = match basis {
        "z" => Basis::Z,
        "x" => Basis::X,
        _ => return Err(err(format!("unknown basis '{basis}'"))),
    };
    Ok(generate(kind, d, rounds, noise, basis).map_err(err)?.to_stim())
}

/// The circuit's detector error model, built by walking it backwards, as
/// Stim's DEM text; `decompose` splits hyperedges into graphlike pieces.
#[pyfunction]
#[pyo3(signature = (circuit_text, decompose=false))]
fn dem_from_circuit(circuit_text: &str, decompose: bool) -> PyResult<String> {
    let c = Circuit::parse(circuit_text).map_err(err)?;
    let dem = if decompose { Dem::from_circuit(&c) } else { Dem::from_circuit_undecomposed(&c) };
    Ok(dem.map_err(err)?.to_stim(decompose))
}

type Decoded<'py> = (Bound<'py, PyBytes>, Bound<'py, PyBytes>, usize, f64);

/// A window as the bindings describe it: (first layer, end layer, commit
/// start, commit end, phase, the windows it waits for), the last as the
/// engine's own scheduler reads it (`stream::dependencies`).
type WindowInfo = (u32, u32, u32, u32, usize, Vec<usize>);

fn decode_packed<'py>(
    py: Python<'py>,
    dem: &Dem,
    packed: &[u8],
    num_shots: usize,
    threads: usize,
    correlated: bool,
) -> PyResult<Decoded<'py>> {
    let decoder = DemDecoder::new(dem).map_err(err)?;
    let graph = decoder.graph();
    // Built only for correlated matching: plain matching never reads them.
    let corr = correlated.then(|| decoder.correlations());
    let nd = dem.num_detectors;
    check_rows(packed.len(), nd.div_ceil(8), num_shots, nd, "detectors")?;
    let start = Instant::now();
    let out = py.allow_threads(|| match_shots(graph, corr, packed, nd, num_shots, threads));
    let seconds = start.elapsed().as_secs_f64();
    let errors = out.iter().filter(|(_, w)| w.is_nan()).count();
    let preds: Vec<u8> = out.iter().flat_map(|(o, _)| o.to_le_bytes()).collect();
    let weights: Vec<u8> = out.iter().flat_map(|(_, w)| w.to_le_bytes()).collect();
    Ok((PyBytes::new_bound(py, &preds), PyBytes::new_bound(py, &weights), errors, seconds))
}

/// Decode with a model someone else wrote, pieces and all (Stim's decomposed
/// DEM). `threads = 0` uses every core; `correlated` decodes as PyMatching's
/// `enable_correlations=True` does.
#[pyfunction]
#[pyo3(signature = (dem_text, packed, num_shots, threads=1, correlated=false))]
fn decode_b8<'py>(
    py: Python<'py>,
    dem_text: &str,
    packed: &[u8],
    num_shots: usize,
    threads: usize,
    correlated: bool,
) -> PyResult<Decoded<'py>> {
    let dem = Dem::parse(dem_text).map_err(err)?;
    decode_packed(py, &dem, packed, num_shots, threads, correlated)
}

/// Decode with this engine's own model of the circuit, and its own
/// decomposition. `threads = 0` uses every core.
#[pyfunction]
#[pyo3(signature = (circuit_text, packed, num_shots, threads=1, correlated=false))]
fn decode_b8_own<'py>(
    py: Python<'py>,
    circuit_text: &str,
    packed: &[u8],
    num_shots: usize,
    threads: usize,
    correlated: bool,
) -> PyResult<Decoded<'py>> {
    let c = Circuit::parse(circuit_text).map_err(err)?;
    let dem = Dem::from_circuit(&c).map_err(err)?;
    decode_packed(py, &dem, packed, num_shots, threads, correlated)
}

/// One model's decoder, shot by shot, for looking inside correlated
/// matching: our first pass's edges, and a second pass from edges given.
/// Edges are `(u, v)` with `-1` for the boundary, as PyMatching gives them.
#[pyclass(unsendable)]
struct Decoder {
    inner: DemDecoder,
    num_detectors: u32,
}

#[pymethods]
impl Decoder {
    #[new]
    fn new(dem_text: &str) -> PyResult<Self> {
        let dem = Dem::parse(dem_text).map_err(err)?;
        let inner = DemDecoder::new(&dem).map_err(err)?;
        Ok(Decoder { inner, num_detectors: dem.num_detectors as u32 })
    }

    fn edges(&self, defects: Vec<u32>) -> PyResult<Vec<(i64, i64)>> {
        let nd = self.num_detectors;
        let edges = self.inner.decode_to_edges(&defects).map_err(|e| err(format!("{e:?}")))?;
        Ok(edges.into_iter().map(|(u, v)| (u as i64, if v == nd { -1 } else { v as i64 })).collect())
    }

    fn pass2(&self, defects: Vec<u32>, edges: Vec<(i64, i64)>) -> PyResult<(u64, f64)> {
        let nd = self.num_detectors;
        let end = |x: i64| match x {
            -1 => Ok(nd),
            x if x >= 0 && x < nd as i64 => Ok(x as u32),
            x => Err(err(format!("{x} is neither a detector nor -1, the boundary"))),
        };
        let edges = edges.into_iter().map(|(u, v)| Ok((end(u)?, end(v)?))).collect::<PyResult<Vec<(u32, u32)>>>()?;
        let p: Prediction = self.inner.decode_pass2(&defects, &edges).map_err(err)?;
        Ok((p.observables, p.weight))
    }
}

/// `num_shots` shots from the frame sampler, one at a time, as b8 rows of
/// detectors and of observables.
#[pyfunction]
fn sample_b8<'py>(
    py: Python<'py>,
    circuit_text: &str,
    num_shots: usize,
    seed: u64,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let c = Circuit::parse(circuit_text).map_err(err)?;
    let sampler = FrameSampler::new(&c).map_err(err)?;
    let mut rng = Xorshift::new(seed);
    let no = sampler.num_observables();
    let mut dets = Vec::new();
    let mut obs = Vec::new();
    for _ in 0..num_shots {
        let shot = sampler.sample(&mut rng);
        pack_row(&shot.detectors, &mut dets);
        let bits: Vec<bool> = (0..no).map(|k| (shot.observables >> k) & 1 == 1).collect();
        pack_row(&bits, &mut obs);
    }
    Ok((PyBytes::new_bound(py, &dets), PyBytes::new_bound(py, &obs)))
}

/// Raw measurements and sweep bits (b8) to detection events and observable
/// flips (b8), as `stim m2d` gives them.
#[pyfunction]
fn m2d_b8<'py>(
    py: Python<'py>,
    circuit_text: &str,
    meas: &[u8],
    sweeps: &[u8],
    num_shots: usize,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let c = Circuit::parse(circuit_text).map_err(err)?;
    let m = crate::m2d::M2d::new(&c).map_err(err)?;
    let (d, o) = py.allow_threads(|| m.convert_b8(meas, sweeps, num_shots)).map_err(err)?;
    Ok((PyBytes::new_bound(py, &d), PyBytes::new_bound(py, &o)))
}

/// A circuit through this engine's parser and printer.
#[pyfunction]
fn circuit_to_stim(text: &str) -> PyResult<String> {
    Ok(Circuit::parse(text).map_err(err)?.to_stim())
}

/// `num_shots` shots from the bit-parallel sampler, 64 at a time, as b8 rows
/// of detectors and of observables, and the seconds it took. `threads = 0`
/// uses every core; each thread samples whole batches from its own stream.
#[pyfunction]
#[pyo3(signature = (circuit_text, num_shots, seed, threads=1))]
fn sample_b8_batch<'py>(
    py: Python<'py>,
    circuit_text: &str,
    num_shots: usize,
    seed: u64,
    threads: usize,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, f64)> {
    let c = Circuit::parse(circuit_text).map_err(err)?;
    let sampler = crate::batch_sampler::BatchSampler::new(&c).map_err(err)?;
    let batches = num_shots.div_ceil(64);
    let threads = resolve_threads(threads, batches);
    let per = batches.div_ceil(threads);
    let start = Instant::now();
    let parts: Vec<(Vec<u8>, Vec<u8>)> = py.allow_threads(|| {
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let sampler = &sampler;
                    scope.spawn(move || {
                        let mut rng = Xorshift::new(seed ^ (t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                        let (mut dets, mut obs) = (Vec::new(), Vec::new());
                        for b in (t * per)..((t + 1) * per).min(batches) {
                            let lanes = (num_shots - b * 64).min(64);
                            sampler.sample(&mut rng).write_b8(lanes, &mut dets, &mut obs);
                        }
                        (dets, obs)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("sampler thread panicked")).collect()
        })
    });
    let seconds = start.elapsed().as_secs_f64();
    let (mut dets, mut obs) = (Vec::new(), Vec::new());
    for (d, o) in parts {
        dets.extend(d);
        obs.extend(o);
    }
    Ok((PyBytes::new_bound(py, &dets), PyBytes::new_bound(py, &obs), seconds))
}

/// Window decoding of b8 shots. Returns the predictions (u64 per shot,
/// u64::MAX where a window failed to decode and the shot was abandoned), the
/// number of defects left unexplained over the shots decoded (zero unless
/// something is wrong), each window's
/// decode time per shot in seconds (f64, shots × windows, when `timings`), and
/// the windows as (first layer, end layer, commit start, commit end, phase,
/// the windows it waits for).
#[pyfunction]
#[pyo3(signature = (dem_text, packed, num_shots, commit, buffer, mode, correlated=false, threads=0, timings=false))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn decode_b8_window<'py>(
    py: Python<'py>,
    dem_text: &str,
    packed: &[u8],
    num_shots: usize,
    commit: usize,
    buffer: usize,
    mode: &str,
    correlated: bool,
    threads: usize,
    timings: bool,
) -> PyResult<(Bound<'py, PyBytes>, usize, Bound<'py, PyBytes>, Vec<WindowInfo>)> {
    use crate::window::{Mode, Model, WindowDecoder};
    let dem = Dem::parse(dem_text).map_err(err)?;
    let mode = match mode {
        "sliding" => Mode::Sliding,
        "parallel" => Mode::Parallel,
        other => return Err(err(format!("mode '{other}' is neither sliding nor parallel"))),
    };
    let wd = WindowDecoder::new(Model::new(&dem).map_err(err)?, commit, buffer, mode).map_err(err)?;
    let nd = dem.num_detectors;
    check_rows(packed.len(), nd.div_ceil(8), num_shots, nd, "detectors")?;
    let out = py.allow_threads(|| window_shots(&wd, packed, nd, num_shots, correlated, threads, timings));
    let preds: Vec<u8> = out.iter().flat_map(|x| x.0.to_le_bytes()).collect();
    let unexplained = out.iter().map(|x| x.1).sum();
    let times: Vec<u8> = out.iter().flat_map(|x| x.2.iter().flat_map(|t| t.to_le_bytes())).collect();
    Ok((PyBytes::new_bound(py, &preds), unexplained, PyBytes::new_bound(py, &times), window_info(&wd)))
}

/// A long SD6 memory decoded as it streams: `batches` × 64 streams of
/// `rounds` rounds of a rotated (or XZZX) patch, sampled by the batch sampler
/// round by round and window-decoded with graphs from a short template.
/// Returns (failures, streams, unexplained defects, lane 0's window decode
/// times in seconds (f64, one per window, per batch), the windows as (first
/// layer, end layer, commit start, commit end, phase, the windows it waits
/// for), wall seconds).
#[pyfunction]
#[pyo3(signature = (code, d, p, rounds, commit, buffer, mode, correlated=false, batches=1, seed=1, threads=0))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn stream_decode<'py>(
    py: Python<'py>,
    code: &str,
    d: usize,
    p: f64,
    rounds: usize,
    commit: usize,
    buffer: usize,
    mode: &str,
    correlated: bool,
    batches: usize,
    seed: u64,
    threads: usize,
) -> PyResult<(usize, usize, usize, Bound<'py, PyBytes>, Vec<WindowInfo>, f64)> {
    use crate::window::Mode;
    let kind = match code {
        "rotated" => CodeKind::Rotated,
        "xzzx" => CodeKind::Xzzx,
        other => return Err(err(format!("unknown code '{other}'"))),
    };
    let mode = match mode {
        "sliding" => Mode::Sliding,
        "parallel" => Mode::Parallel,
        other => return Err(err(format!("mode '{other}' is neither sliding nor parallel"))),
    };
    let out = py.allow_threads(|| crate::batch::stream_shots(kind, d, p, rounds, commit, buffer, mode, correlated, batches, seed, threads)).map_err(err)?;
    let bytes: Vec<u8> = out.times.iter().flat_map(|x| x.to_le_bytes()).collect();
    Ok((out.failures, out.shots, out.unexplained, PyBytes::new_bound(py, &bytes), out.windows, out.seconds))
}

fn bp_method(method: &str, ms_scale: f64) -> PyResult<crate::bp::Method> {
    match method {
        "product_sum" => Ok(crate::bp::Method::ProductSum),
        "minimum_sum" => Ok(crate::bp::Method::MinSum { scale: ms_scale }),
        other => Err(err(format!("BP method '{other}' is neither product_sum nor minimum_sum"))),
    }
}

/// Belief propagation on a parity-check matrix given by its columns (the
/// checks each variable touches), flooding schedule, as `ldpc` runs it.
/// Returns (hard decision, posterior log-likelihood ratios, converged,
/// iterations).
#[pyfunction]
#[pyo3(signature = (num_checks, columns, priors, syndrome, max_iter=20, method="product_sum", ms_scale=1.0))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn bp_decode(
    num_checks: usize,
    columns: Vec<Vec<u32>>,
    priors: Vec<f64>,
    syndrome: Vec<u8>,
    max_iter: usize,
    method: &str,
    ms_scale: f64,
) -> PyResult<(Vec<u8>, Vec<f64>, bool, usize)> {
    let bp = crate::bp::Bp::new(num_checks, &columns, &priors).map_err(err)?;
    if syndrome.len() != num_checks {
        return Err(err(format!("{} syndrome bits for {num_checks} checks", syndrome.len())));
    }
    let mut w = bp.work();
    let out = bp.decode(&syndrome, bp_method(method, ms_scale)?, max_iter, &mut w);
    Ok((w.hard, w.llr, out.converged, out.iterations))
}

/// Belief-matching of b8 shots on a decomposed model, as `beliefmatching`
/// decodes: BP on the hypergraph, its own correction where it converges, and
/// otherwise matching on weights −ln p from its posteriors. Returns
/// (predictions as u64 per shot, u64::MAX where decoding failed; the matching's
/// weight per shot as f64, NaN where BP converged; one byte per shot, 1 where
/// BP converged; shots that failed; seconds).
#[pyfunction]
#[pyo3(signature = (dem_text, packed, num_shots, max_iter=20, method="product_sum", ms_scale=1.0, threads=0))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn decode_b8_belief<'py>(
    py: Python<'py>,
    dem_text: &str,
    packed: &[u8],
    num_shots: usize,
    max_iter: usize,
    method: &str,
    ms_scale: f64,
    threads: usize,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, Bound<'py, PyBytes>, usize, f64)> {
    use crate::belief::BeliefMatching;
    let dem = Dem::parse(dem_text).map_err(err)?;
    let bm = BeliefMatching::from_dem(&dem, bp_method(method, ms_scale)?, max_iter).map_err(err)?;
    let nd = dem.num_detectors;
    check_rows(packed.len(), nd.div_ceil(8), num_shots, nd, "detectors")?;
    let start = Instant::now();
    let out = py.allow_threads(|| belief_shots(&bm, packed, nd, num_shots, threads));
    let seconds = start.elapsed().as_secs_f64();
    let preds: Vec<u8> = out.iter().flat_map(|x| x.0.to_le_bytes()).collect();
    let weights: Vec<u8> = out.iter().flat_map(|x| x.1.to_le_bytes()).collect();
    let conv: Vec<u8> = out.iter().map(|x| u8::from(x.2 == 1)).collect();
    let errors = out.iter().filter(|x| x.2 == 2).count();
    Ok((PyBytes::new_bound(py, &preds), PyBytes::new_bound(py, &weights), PyBytes::new_bound(py, &conv), errors, seconds))
}

fn bb_cycles(total: usize, least: usize) -> PyResult<()> {
    crate::bb::check_cycles(total, least).map_err(err)
}

fn bb_code(name: &str) -> PyResult<crate::bb::BbCode> {
    match name {
        "gross" | "144" => Ok(crate::bb::BbCode::gross()),
        "72" => Ok(crate::bb::BbCode::bb72()),
        other => Err(err(format!("unknown bivariate bicycle code '{other}' (gross or 72)"))),
    }
}

/// A bivariate bicycle code's check matrices and paired logical operators,
/// each as a list of rows, a row being the data qubits it acts on:
/// (H_X, H_Z, logical X, logical Z).
#[pyfunction]
#[allow(clippy::type_complexity)]
fn bb_matrices(code: &str) -> PyResult<(Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<Vec<usize>>)> {
    let c = bb_code(code)?;
    let rows = |m: &crate::gf2::BitMatrix| (0..m.rows).map(|r| m.row_ones(r)).collect::<Vec<_>>();
    let (lx, lz) = c.logicals();
    Ok((rows(&c.hx()), rows(&c.hz()), rows(&lx), rows(&lz)))
}

/// The paper's Z-basis memory on a bivariate bicycle code ("gross" or "72"),
/// `cycles` depth-8 syndrome cycles under circuit noise `p`, as Stim text.
#[pyfunction]
fn bb_memory_circuit(code: &str, cycles: usize, p: f64) -> PyResult<String> {
    bb_cycles(cycles, cycles)?;
    crate::memory::probability(p).map_err(err)?;
    Ok(bb_code(code)?.memory_z(cycles, p))
}

/// Every automorphism of a bivariate bicycle code, a shift x^a y^b of both
/// halves with or without the ZX-duality, and its action on the logical
/// qubits: (a, b, dual, the 24 × 24 matrix's rows as column lists; row i
/// is the image of basis element i, X logicals then Z logicals, in the
/// basis `bb_matrices` gives).
#[pyfunction]
#[allow(clippy::type_complexity)]
fn bb_automorphisms(code: &str) -> PyResult<Vec<(usize, usize, bool, Vec<Vec<usize>>)>> {
    use crate::bb_auto::Automorphism;
    let c = bb_code(code)?;
    let mut out = Vec::new();
    for a in 0..c.l {
        for b in 0..c.m {
            for dual in [false, true] {
                let m = c.logical_action(Automorphism { shift: (a, b), dual }).map_err(err)?;
                out.push((a, b, dual, (0..m.rows).map(|r| m.row_ones(r)).collect()));
            }
        }
    }
    Ok(out)
}

fn gauged(operator: &str, expanded: bool) -> PyResult<(crate::bb::BbCode, crate::bb_gauge::Gauging)> {
    let code = crate::bb::BbCode::gross();
    let support = crate::bb::gross_operator(operator).map_err(err)?;
    let g = if expanded { crate::bb_gauge::Gauging::expanded(&code, &support) } else { crate::bb_gauge::Gauging::new(&code, &support) };
    Ok((code, g.map_err(err)?))
}

/// The gauging ancilla system that measures a gross-code logical X
/// ("f" = X(f, 0), "gh" = X(g, h), "f+gh" their product): Cross et al.'s
/// minimal system, or with `expanded` edges added until its Cheeger constant
/// is at least 1. Returns (support, edges (the Z checks touching it), the
/// added edges as vertex pairs, each edge's ends (indices into support), each
/// vertex's edges, the flux checks (edge indices), the deformed H_X' and H_Z'
/// as rows over data then edge qubits, the merged cycle's ticks, the worst
/// cut (|δU|, |U|)).
#[pyfunction]
#[pyo3(signature = (operator, expanded=false))]
#[allow(clippy::type_complexity)]
fn bb_gauging(
    operator: &str,
    expanded: bool,
) -> PyResult<(
    Vec<usize>,
    Vec<usize>,
    Vec<(usize, usize)>,
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
    usize,
    (usize, usize),
)> {
    let (code, g) = gauged(operator, expanded)?;
    let (hx, hz) = g.deformed(&code);
    let rows = |m: &crate::gf2::BitMatrix| (0..m.rows).map(|r| m.row_ones(r)).collect::<Vec<_>>();
    let ticks = crate::bb_circuit::merged_cycle(&code, &g).ticks.len();
    let (b, u) = g.cheeger();
    Ok((
        g.support.clone(),
        g.edges.clone(),
        g.extra.clone(),
        g.incidence.clone(),
        g.gauss.clone(),
        g.flux.clone(),
        rows(&hx),
        rows(&hz),
        ticks,
        (b, u.len()),
    ))
}

/// A memory of a bivariate bicycle code in either basis, by the cycle
/// writer (in "z" it is `bb_memory_circuit`'s model), as Stim text.
#[pyfunction]
fn bb_memory_basis_circuit(code: &str, basis: &str, cycles: usize, p: f64) -> PyResult<String> {
    bb_cycles(cycles, cycles)?;
    crate::memory::probability(p).map_err(err)?;
    Ok(crate::bb_circuit::memory(&bb_code(code)?, basis_of(basis)?, cycles, p).to_stim())
}

/// The gauging measurement of a gross-code logical X (see `bb_gauging`) as
/// Stim text: `pre` memory cycles, `merged` cycles of the deformed code,
/// `post` memory cycles, the data read in `basis`. "x": L0 the outcome,
/// L1-L12 the X logicals; "z": the 11 Z logicals that commute with it.
/// `expanded` uses the expanded ancilla system (see `bb_gauging`).
#[pyfunction]
#[pyo3(signature = (operator, basis, pre, merged, post, p, expanded=false))]
fn bb_logical_measurement_circuit(operator: &str, basis: &str, pre: usize, merged: usize, post: usize, p: f64, expanded: bool) -> PyResult<String> {
    // A merge of no cycles is refused below, with its own message.
    bb_cycles(pre.saturating_add(merged).saturating_add(post).max(1), merged.max(1))?;
    crate::memory::probability(p).map_err(err)?;
    let (code, g) = gauged(operator, expanded)?;
    let c = crate::bb_circuit::logical_measurement(&code, &g, basis_of(basis)?, pre, merged, post, p).map_err(err)?;
    Ok(c.to_stim())
}

fn osd_method(name: &str, order: usize) -> PyResult<crate::osd::OsdMethod> {
    use crate::osd::OsdMethod;
    match name {
        "osd_0" | "osd0" => Ok(OsdMethod::Osd0),
        "osd_e" => Ok(OsdMethod::Exhaustive(order)),
        "osd_cs" => Ok(OsdMethod::CombinationSweep(order)),
        other => Err(err(format!("OSD method '{other}' is not osd_0, osd_e or osd_cs"))),
    }
}

/// BP+OSD on a parity-check matrix given by its columns. Returns (correction,
/// BP converged, iterations).
#[pyfunction]
#[pyo3(signature = (num_checks, columns, priors, syndrome, max_iter=20, method="minimum_sum", ms_scale=0.0, osd="osd_cs", osd_order=7))]
#[allow(clippy::too_many_arguments)]
fn bposd_decode(
    num_checks: usize,
    columns: Vec<Vec<u32>>,
    priors: Vec<f64>,
    syndrome: Vec<u8>,
    max_iter: usize,
    method: &str,
    ms_scale: f64,
    osd: &str,
    osd_order: usize,
) -> PyResult<(Vec<u8>, bool, usize)> {
    use crate::osd::BpOsd;
    let dec = BpOsd::new(num_checks, columns, &priors, bp_method(method, ms_scale)?, max_iter, osd_method(osd, osd_order)?)
        .map_err(err)?;
    if syndrome.len() != num_checks {
        return Err(err(format!("{} syndrome bits for {num_checks} checks", syndrome.len())));
    }
    let mut w = dec.work();
    let out = dec.decode(&syndrome, &mut w);
    Ok((w.correction, out.converged, out.iterations))
}

/// BP+OSD of b8 shots on an undecomposed error model: each fault a column,
/// its prior the model's. Returns (predictions as u64 per shot; one byte per
/// shot, 1 where BP converged; seconds).
#[pyfunction]
#[pyo3(signature = (dem_text, packed, num_shots, max_iter=10_000, method="minimum_sum", ms_scale=0.0, osd="osd_cs", osd_order=7, threads=0))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn decode_b8_bposd<'py>(
    py: Python<'py>,
    dem_text: &str,
    packed: &[u8],
    num_shots: usize,
    max_iter: usize,
    method: &str,
    ms_scale: f64,
    osd: &str,
    osd_order: usize,
    threads: usize,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>, f64)> {
    use crate::osd::BpOsd;
    let dem = Dem::parse(dem_text).map_err(err)?;
    let columns: Vec<Vec<u32>> = dem.mechanisms.iter().map(|m| m.detectors.clone()).collect();
    let priors: Vec<f64> = dem.mechanisms.iter().map(|m| m.p).collect();
    let obs: Vec<u64> = dem.mechanisms.iter().map(|m| m.observables).collect();
    let dec = BpOsd::new(dem.num_detectors, columns, &priors, bp_method(method, ms_scale)?, max_iter, osd_method(osd, osd_order)?)
        .map_err(err)?;
    let nd = dem.num_detectors;
    check_rows(packed.len(), nd.div_ceil(8), num_shots, nd, "detectors")?;
    let start = Instant::now();
    let out = py.allow_threads(|| bposd_shots(&dec, &obs, packed, nd, num_shots, threads));
    let seconds = start.elapsed().as_secs_f64();
    let preds: Vec<u8> = out.iter().flat_map(|x| x.0.to_le_bytes()).collect();
    let conv: Vec<u8> = out.iter().map(|x| x.1).collect();
    Ok((PyBytes::new_bound(py, &preds), PyBytes::new_bound(py, &conv), seconds))
}

/// Lattice surgery as a circuit (Stim text): two rotated distance-`d` patches,
/// `pre` rounds apart, the seam prepared in |+> and `merged` rounds of the
/// merged patch, the split, `post` rounds apart, and the readout, SD6 noise
/// `p`. `basis` "z": both patches in |0>, observables L0 = the merge outcome
/// Z1Z2, L1 = Z1, L2 = Z2. "x": both in |+>, L0 = X1X2.
#[pyfunction]
#[pyo3(signature = (d, merged, p, basis="z", pre=None, post=None))]
fn surgery_circuit(d: usize, merged: usize, p: f64, basis: &str, pre: Option<usize>, post: Option<usize>) -> PyResult<String> {
    let basis = basis_of(basis)?;
    Ok(crate::surgery::zz_circuit(d, pre.unwrap_or(d), merged, post.unwrap_or(d), p, basis).map_err(err)?.to_stim())
}

fn basis_of(name: &str) -> PyResult<Basis> {
    match name {
        "z" => Ok(Basis::Z),
        "x" => Ok(Basis::X),
        other => Err(err(format!("unknown basis '{other}'"))),
    }
}

/// A logical CNOT by lattice surgery (see `surgery::cnot`), as Stim text:
/// control, an ancilla in |+> and target in an L; Z_C Z_A, then X_A X_T, then
/// the ancilla read in Z. `inputs` "z": |0>|0>, observables L0 = Z_C and L1 =
/// Z_T with its frame. "x": |+>|+>, L0 = X_T and L1 = X_C X_T with its frame.
#[pyfunction]
#[pyo3(signature = (d, merged, p, inputs="z"))]
fn surgery_cnot(d: usize, merged: usize, p: f64, inputs: &str) -> PyResult<String> {
    Ok(crate::surgery::cnot_circuit(d, merged, p, basis_of(inputs)?).map_err(err)?.to_stim())
}

/// k Z⊗Z measurements in a row on two patches in |0>|0> (see
/// `surgery::repeated`): observables each outcome, then Z1 and Z2.
#[pyfunction]
fn surgery_repeated(d: usize, k: usize, merged: usize, p: f64) -> PyResult<String> {
    Ok(crate::surgery::repeated_circuit(d, k, merged, p).map_err(err)?.to_stim())
}

/// n patches in a row, all in |0>, merged at once (see `surgery::line`):
/// each neighbouring pair's Z⊗Z, n − 1 parities. Observables each seam's
/// outcome, then each patch's Z.
#[pyfunction]
fn surgery_line(d: usize, n: usize, merged: usize, p: f64) -> PyResult<String> {
    Ok(crate::surgery::line_circuit(d, n, merged, p).map_err(err)?.to_stim())
}

/// The X⊗X mirror of `surgery_circuit` (see `surgery::vertical`): "x", the
/// outcome and each patch's X; "z", Z1Z2 with its seam qubit.
#[pyfunction]
#[pyo3(signature = (d, merged, p, basis="x"))]
fn surgery_vertical(d: usize, merged: usize, p: f64, basis: &str) -> PyResult<String> {
    Ok(crate::surgery::xx_circuit(d, merged, p, basis_of(basis)?).map_err(err)?.to_stim())
}

/// b8 rows of `num_bits` bits as Stim's 01 text.
#[pyfunction]
fn b8_to_01(packed: &[u8], num_bits: usize) -> PyResult<String> {
    Ok(write_01(&read_b8(packed, num_bits).map_err(err)?))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(dem_from_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(decode_b8, m)?)?;
    m.add_function(wrap_pyfunction!(decode_b8_own, m)?)?;
    m.add_function(wrap_pyfunction!(sample_b8, m)?)?;
    m.add_function(wrap_pyfunction!(b8_to_01, m)?)?;
    m.add_class::<Decoder>()?;
    m.add_function(wrap_pyfunction!(m2d_b8, m)?)?;
    m.add_function(wrap_pyfunction!(circuit_to_stim, m)?)?;
    m.add_function(wrap_pyfunction!(sample_b8_batch, m)?)?;
    m.add_function(wrap_pyfunction!(decode_b8_window, m)?)?;
    m.add_function(wrap_pyfunction!(stream_decode, m)?)?;
    m.add_function(wrap_pyfunction!(bp_decode, m)?)?;
    m.add_function(wrap_pyfunction!(decode_b8_belief, m)?)?;
    m.add_function(wrap_pyfunction!(bb_matrices, m)?)?;
    m.add_function(wrap_pyfunction!(bb_memory_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(bposd_decode, m)?)?;
    m.add_function(wrap_pyfunction!(decode_b8_bposd, m)?)?;
    m.add_function(wrap_pyfunction!(bb_automorphisms, m)?)?;
    m.add_function(wrap_pyfunction!(bb_gauging, m)?)?;
    m.add_function(wrap_pyfunction!(bb_memory_basis_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(bb_logical_measurement_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(surgery_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(surgery_cnot, m)?)?;
    m.add_function(wrap_pyfunction!(surgery_repeated, m)?)?;
    m.add_function(wrap_pyfunction!(surgery_line, m)?)?;
    m.add_function(wrap_pyfunction!(surgery_vertical, m)?)?;
    Ok(())
}

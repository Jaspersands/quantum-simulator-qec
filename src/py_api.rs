//! Python bindings for the cross-check harness (`tools/xcheck.py`).
//!
//! Shots cross the boundary as bytes in Stim's b8 layout (numpy's
//! `packbits(..., bitorder="little")`), and predictions come back as
//! little-endian u64 and f64 arrays, so no numpy crate is needed on this side.

use std::time::Instant;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::circuit::{Basis, Circuit};
use crate::dem::Dem;
use crate::dem_decoder::{DemDecoder, Prediction};
use crate::frame_sampler::FrameSampler;
use crate::memory::{generate, CodeKind, NoiseModel};
use crate::shots::{pack_row, read_b8, write_01};
use crate::sparse::Scratch;
use crate::surface_code::Xorshift;

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

/// `threads = 0` means every core; never more threads than units of work.
fn resolve_threads(threads: usize, units: usize) -> usize {
    let threads = if threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { threads };
    threads.clamp(1, units.max(1))
}

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
    Ok(Dem::from_circuit(&c).map_err(err)?.to_stim(decompose))
}

type Decoded<'py> = (Bound<'py, PyBytes>, Bound<'py, PyBytes>, usize, f64);

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
    let corr = decoder.correlations();
    let nd = dem.num_detectors;
    let stride = nd.div_ceil(8);
    if packed.len() != stride * num_shots {
        return Err(err(format!("{} bytes is not {num_shots} shots of {nd} detectors", packed.len())));
    }
    let threads = resolve_threads(threads, num_shots);
    let chunk = num_shots.div_ceil(threads);
    let start = Instant::now();
    let parts: Vec<Vec<(u64, f64)>> = py.allow_threads(|| {
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let (lo, hi) = (t * chunk, ((t + 1) * chunk).min(num_shots));
                    scope.spawn(move || {
                        let mut scratch = Scratch::new(graph);
                        let mut defects = Vec::new();
                        let mut out = Vec::with_capacity(hi.saturating_sub(lo));
                        for s in lo..hi {
                            defects.clear();
                            let row = &packed[s * stride..(s + 1) * stride];
                            for i in 0..nd {
                                if (row[i / 8] >> (i % 8)) & 1 == 1 {
                                    defects.push(i as u32);
                                }
                            }
                            // A failed shot is written as all-ones observables and a NaN
                            // weight, and counted by the NaN: all-ones is a real prediction
                            // when a model has 64 observables.
                            let result = if correlated {
                                graph.decode_correlated(corr, &mut scratch, &defects)
                            } else {
                                graph.decode(&mut scratch, &defects)
                            };
                            out.push(match result {
                                Ok(p) => (p.observables, p.weight),
                                Err(_) => (u64::MAX, f64::NAN),
                            });
                        }
                        out
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("decoder thread panicked")).collect()
        })
    });
    let seconds = start.elapsed().as_secs_f64();
    let mut preds = Vec::with_capacity(8 * num_shots);
    let mut weights = Vec::with_capacity(8 * num_shots);
    let mut errors = 0usize;
    for (o, w) in parts.into_iter().flatten() {
        errors += usize::from(w.is_nan());
        preds.extend_from_slice(&o.to_le_bytes());
        weights.extend_from_slice(&w.to_le_bytes());
    }
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
/// the windows as (first layer, end layer, commit start, commit end, phase).
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
) -> PyResult<(Bound<'py, PyBytes>, usize, Bound<'py, PyBytes>, Vec<(u32, u32, u32, u32, usize)>)> {
    use crate::window::{Mode, Model, WindowDecoder};
    let dem = Dem::parse(dem_text).map_err(err)?;
    let mode = match mode {
        "sliding" => Mode::Sliding,
        "parallel" => Mode::Parallel,
        other => return Err(err(format!("mode '{other}' is neither sliding nor parallel"))),
    };
    let wd = WindowDecoder::new(Model::new(&dem).map_err(err)?, commit, buffer, mode).map_err(err)?;
    let nd = dem.num_detectors;
    let stride = nd.div_ceil(8);
    if packed.len() != stride * num_shots {
        return Err(err(format!("{} bytes is not {num_shots} shots of {nd} detectors", packed.len())));
    }
    let nw = wd.windows.len();
    let threads = resolve_threads(threads, num_shots);
    let chunk = num_shots.div_ceil(threads);
    let parts: Vec<(Vec<u64>, usize, Vec<f64>)> = py.allow_threads(|| {
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let wd = &wd;
                    scope.spawn(move || {
                        let mut scratches = wd.scratches();
                        let (mut preds, mut unexplained, mut times) = (Vec::new(), 0usize, Vec::new());
                        let mut live = vec![false; nd];
                        for s in (t * chunk)..((t + 1) * chunk).min(num_shots) {
                            let row = &packed[s * stride..(s + 1) * stride];
                            for (i, l) in live.iter_mut().enumerate() {
                                *l = (row[i / 8] >> (i % 8)) & 1 == 1;
                            }
                            let mut obs = 0u64;
                            let mut failed = false;
                            let mut shot_times = vec![0.0; if timings { nw } else { 0 }];
                            'windows: for phase in &wd.phases {
                                for &wi in phase {
                                    let start = timings.then(Instant::now);
                                    if wd.decode_window(wi, &mut live, &mut obs, correlated, &mut scratches[wi]).is_err() {
                                        // A refused window fails the shot; its defects are
                                        // not "unexplained", which counts commit mistakes.
                                        failed = true;
                                        break 'windows;
                                    }
                                    if let Some(start) = start {
                                        shot_times[wi] = start.elapsed().as_secs_f64();
                                    }
                                }
                            }
                            if !failed {
                                unexplained += live.iter().filter(|&&b| b).count();
                            }
                            preds.push(if failed { u64::MAX } else { obs });
                            times.extend(shot_times);
                        }
                        (preds, unexplained, times)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("window thread panicked")).collect()
        })
    });
    let (mut preds, mut unexplained, mut times) = (Vec::new(), 0usize, Vec::new());
    for (p, u, t) in parts {
        preds.extend(p.iter().flat_map(|x| x.to_le_bytes()));
        unexplained += u;
        times.extend(t.iter().flat_map(|x| x.to_le_bytes()));
    }
    let mut phase_of = vec![0usize; nw];
    for (k, phase) in wd.phases.iter().enumerate() {
        for &wi in phase {
            phase_of[wi] = k;
        }
    }
    let info = wd
        .windows
        .iter()
        .enumerate()
        .map(|(i, w)| (w.window.layers.0, w.window.layers.1, w.commit.0, w.commit.1, phase_of[i]))
        .collect();
    Ok((PyBytes::new_bound(py, &preds), unexplained, PyBytes::new_bound(py, &times), info))
}

/// A long SD6 memory decoded as it streams: `batches` × 64 streams of
/// `rounds` rounds of a rotated (or XZZX) patch, sampled by the batch sampler
/// round by round and window-decoded with graphs from a short template.
/// Returns (failures, streams, unexplained defects, lane 0's window decode
/// times in seconds (f64, one per window, per batch), the windows as (first
/// layer, end layer, commit start, commit end, phase), wall seconds).
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
) -> PyResult<(usize, usize, usize, Bound<'py, PyBytes>, Vec<(u32, u32, u32, u32, usize)>, f64)> {
    use crate::stream::{run_stream, StreamDecoder};
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
    let dec = StreamDecoder::new(kind, d, p, Basis::Z, commit, buffer, mode, rounds).map_err(err)?;
    let plan = dec.plan(rounds).map_err(err)?;
    let circuit = crate::memory::generate_repeat(kind, d, rounds, p, Basis::Z).map_err(err)?;
    let sampler = crate::batch_sampler::BatchSampler::new(&circuit).map_err(err)?;
    let threads = resolve_threads(threads, batches);
    let start = Instant::now();
    let parts: Vec<Result<(usize, usize, Vec<f64>), String>> = py.allow_threads(|| {
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let (dec, plan, sampler) = (&dec, &plan, &sampler);
                    scope.spawn(move || {
                        let clock_origin = Instant::now();
                        let clock = move || clock_origin.elapsed().as_secs_f64();
                        let mut scratches = dec.scratches();
                        let (mut failures, mut unexplained, mut times) = (0usize, 0usize, Vec::new());
                        for b in (t..batches).step_by(threads) {
                            let mut rng = Xorshift::new(seed ^ (b as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                            let (pred, truth, u, ts) =
                                run_stream(dec, plan, sampler, &mut rng, correlated, &mut scratches, Some(&clock))
                                    .map_err(|e| format!("{e:?}"))?;
                            failures += pred.iter().enumerate().filter(|(lane, &p)| ((p ^ (truth >> lane)) & 1) == 1).count();
                            unexplained += u;
                            times.extend(ts);
                        }
                        Ok((failures, unexplained, times))
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("stream thread panicked")).collect()
        })
    });
    let seconds = start.elapsed().as_secs_f64();
    let (mut failures, mut unexplained, mut times) = (0usize, 0usize, Vec::new());
    for part in parts {
        let (f, u, t) = part.map_err(err)?;
        failures += f;
        unexplained += u;
        times.extend(t);
    }
    let info = plan.specs.iter().map(|s| (s.a, s.b, s.commit.0, s.commit.1, s.phase)).collect();
    let bytes: Vec<u8> = times.iter().flat_map(|x| x.to_le_bytes()).collect();
    Ok((failures, batches * 64, unexplained, PyBytes::new_bound(py, &bytes), info, seconds))
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
    Ok(())
}

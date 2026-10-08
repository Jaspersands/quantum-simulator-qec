//! The loops over shots that the Python bindings and the crate's API share: each decoder over
//! Stim's b8 rows, cut across threads, results in shot order.

use crate::belief::BeliefMatching;
use crate::lsd::BpLsd;
use crate::osd::BpOsd;
use crate::relay::Relay;
use crate::parallel::parallel;
use crate::sparse::{Correlations, Scratch, SparseGraph};
use crate::window::WindowDecoder;

/// Matching of b8 shots: (observables, weight) per shot, NaN weight where no matching exists.
pub fn match_shots(
    graph: &SparseGraph,
    corr: Option<&Correlations>,
    packed: &[u8],
    nd: usize,
    num_shots: usize,
    threads: usize,
) -> Vec<(u64, f64)> {
    let stride = nd.div_ceil(8);
    parallel(num_shots, threads, |range| {
        let mut scratch = Scratch::new(graph);
        let mut defects = Vec::new();
        range
            .map(|s| {
                defects.clear();
                crate::shots::defects_from_b8(&packed[s * stride..(s + 1) * stride], nd, &mut defects);
                let result = match corr {
                    Some(corr) => graph.decode_correlated(corr, &mut scratch, &defects),
                    None => graph.decode(&mut scratch, &defects),
                };
                // All-ones is a real prediction when a model has 64 observables: the NaN marks failure.
                result.map_or((u64::MAX, f64::NAN), |p| (p.observables, p.weight))
            })
            .collect()
    })
}

/// Belief-matching of b8 shots: (observables, weight, 0 matched / 1 BP converged / 2 failed).
pub fn belief_shots(bm: &BeliefMatching, packed: &[u8], nd: usize, num_shots: usize, threads: usize) -> Vec<(u64, f64, u8)> {
    let stride = nd.div_ceil(8);
    parallel(num_shots, threads, |range| {
        let mut work = bm.work();
        let mut defects = Vec::new();
        range
            .map(|s| {
                defects.clear();
                crate::shots::defects_from_b8(&packed[s * stride..(s + 1) * stride], nd, &mut defects);
                bm.decode(&defects, &mut work).map_or((u64::MAX, f64::NAN, 2), |o| (o.observables, o.weight, u8::from(o.converged)))
            })
            .collect()
    })
}

/// A decoder of syndromes over a model's faults: the faults it sets are XORed into the
/// predicted observables.
pub trait SyndromeDecoder: Sync {
    type Work;
    fn work(&self) -> Self::Work;
    /// Decode one syndrome (one byte per detector): 1 where BP converged by itself, 0 where
    /// the post-processing ran, 2 where no correction explains the syndrome.
    fn decode_syndrome(&self, syndrome: &[u8], work: &mut Self::Work) -> u8;
    fn correction<'w>(&self, work: &'w Self::Work) -> &'w [u8];
}

impl SyndromeDecoder for BpOsd {
    type Work = crate::osd::BpOsdWork;
    fn work(&self) -> Self::Work {
        BpOsd::work(self)
    }
    fn decode_syndrome(&self, syndrome: &[u8], work: &mut Self::Work) -> u8 {
        u8::from(self.decode(syndrome, work).converged)
    }
    fn correction<'w>(&self, work: &'w Self::Work) -> &'w [u8] {
        &work.correction
    }
}

impl SyndromeDecoder for BpLsd {
    type Work = crate::lsd::BpLsdWork;
    fn work(&self) -> Self::Work {
        BpLsd::work(self)
    }
    fn decode_syndrome(&self, syndrome: &[u8], work: &mut Self::Work) -> u8 {
        let o = self.decode(syndrome, work);
        if !o.solved { 2 } else { u8::from(o.converged) }
    }
    fn correction<'w>(&self, work: &'w Self::Work) -> &'w [u8] {
        &work.correction
    }
}

/// A syndrome decoder over b8 shots on a model's faults: (observables, its flag).
pub fn syndrome_shots<D: SyndromeDecoder>(dec: &D, obs: &[u64], packed: &[u8], nd: usize, num_shots: usize, threads: usize) -> Vec<(u64, u8)> {
    let stride = nd.div_ceil(8);
    parallel(num_shots, threads, |range| {
        let mut work = dec.work();
        let mut syndrome = vec![0u8; nd];
        range
            .map(|s| {
                let row = &packed[s * stride..(s + 1) * stride];
                for (i, x) in syndrome.iter_mut().enumerate() {
                    *x = (row[i / 8] >> (i % 8)) & 1;
                }
                let flag = dec.decode_syndrome(&syndrome, &mut work);
                let pred = dec.correction(&work).iter().zip(obs).filter(|(c, _)| **c != 0).fold(0u64, |a, (_, o)| a ^ o);
                (pred, flag)
            })
            .collect()
    })
}

/// Relay-BP of b8 shots on a model's faults: (observables, a leg converged). Shot `s` draws
/// its memory strengths from a generator seeded by the configured seed and `s`, so a batch's
/// results do not depend on how it is cut across threads.
pub fn relay_shots(dec: &Relay, obs: &[u64], packed: &[u8], nd: usize, num_shots: usize, threads: usize) -> Vec<(u64, u8)> {
    let stride = nd.div_ceil(8);
    parallel(num_shots, threads, |range| {
        let mut work = dec.work();
        let mut syndrome = vec![0u8; nd];
        range
            .map(|s| {
                let row = &packed[s * stride..(s + 1) * stride];
                for (i, x) in syndrome.iter_mut().enumerate() {
                    *x = (row[i / 8] >> (i % 8)) & 1;
                }
                dec.reseed(&mut work, shot_seed(dec.config.seed, s as u64));
                let o = dec.decode(&syndrome, &mut work);
                let pred = work.correction.iter().zip(obs).filter(|(c, _)| **c != 0).fold(0u64, |a, (_, o)| a ^ o);
                (pred, u8::from(o.converged))
            })
            .collect()
    })
}

/// A seed for shot `s` of a batch: SplitMix64's mix of the two.
fn shot_seed(seed: u64, s: u64) -> u64 {
    let mut z = seed.wrapping_add(s.wrapping_add(1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// BP+OSD of b8 shots on a model's faults: (observables, BP converged).
pub fn bposd_shots(dec: &BpOsd, obs: &[u64], packed: &[u8], nd: usize, num_shots: usize, threads: usize) -> Vec<(u64, u8)> {
    syndrome_shots(dec, obs, packed, nd, num_shots, threads)
}

/// Window decoding of b8 shots: per shot (observables or u64::MAX where a window refused, the
/// defects its commits left unexplained, each window's decode seconds when `timings`, whether
/// a window refused).
pub fn window_shots(
    wd: &WindowDecoder,
    packed: &[u8],
    nd: usize,
    num_shots: usize,
    correlated: bool,
    threads: usize,
    timings: bool,
) -> Vec<WindowShot> {
    let stride = nd.div_ceil(8);
    let nw = wd.windows.len();
    parallel(num_shots, threads, |range| {
        let mut scratches = wd.scratches();
        let mut live = vec![false; nd];
        range
            .map(|s| {
                let row = &packed[s * stride..(s + 1) * stride];
                for (i, l) in live.iter_mut().enumerate() {
                    *l = (row[i / 8] >> (i % 8)) & 1 == 1;
                }
                let (mut obs, mut failed) = (0u64, false);
                let mut times = vec![0.0; if timings { nw } else { 0 }];
                'windows: for phase in &wd.phases {
                    for &wi in phase {
                        let start = timings.then(std::time::Instant::now);
                        // A refused window fails the shot; its defects are not "unexplained",
                        // which counts commit mistakes.
                        if wd.decode_window(wi, &mut live, &mut obs, correlated, &mut scratches[wi]).is_err() {
                            failed = true;
                            break 'windows;
                        }
                        if let Some(start) = start {
                            times[wi] = start.elapsed().as_secs_f64();
                        }
                    }
                }
                let unexplained = if failed { 0 } else { live.iter().filter(|&&b| b).count() };
                (if failed { u64::MAX } else { obs }, unexplained, times, failed)
            })
            .collect()
    })
}

/// A window as the bindings describe it: (first layer, end layer, commit start, commit end,
/// phase, the windows it waits for).
pub type WindowInfo = (u32, u32, u32, u32, usize, Vec<usize>);

pub fn window_info(wd: &WindowDecoder) -> Vec<WindowInfo> {
    let mut phase_of = vec![0usize; wd.windows.len()];
    for (k, phase) in wd.phases.iter().enumerate() {
        for &wi in phase {
            phase_of[wi] = k;
        }
    }
    let deps = crate::stream::dependencies(&wd.specs, wd.mode);
    wd.windows
        .iter()
        .zip(deps)
        .enumerate()
        .map(|(i, (w, deps))| (w.window.layers.0, w.window.layers.1, w.commit.0, w.commit.1, phase_of[i], deps))
        .collect()
}


/// A streamed memory's outcome (see `stream_shots`).
pub struct StreamOutcome {
    pub failures: usize,
    pub shots: usize,
    /// Defects a window's commits left unexplained: an engine bug if ever nonzero.
    pub unexplained: usize,
    /// The first stream of each batch's per-window decode seconds, batch after batch.
    pub times: Vec<f64>,
    pub windows: Vec<WindowInfo>,
    pub seconds: f64,
}

/// An SD6 memory of `rounds` rounds, sampled round by round and window-decoded as it streams,
/// in `batches` batches of 64 shots across `threads`: each batch seeded from `seed` and its
/// index, so the failures do not depend on the threads.
#[allow(clippy::too_many_arguments)]
pub fn stream_shots(
    kind: crate::memory::CodeKind,
    d: usize,
    p: f64,
    rounds: usize,
    commit: usize,
    buffer: usize,
    mode: crate::window::Mode,
    correlated: bool,
    batches: usize,
    seed: u64,
    threads: usize,
) -> Result<StreamOutcome, String> {
    use crate::circuit::Basis;
    use crate::stream::StreamDecoder;
    let dec = StreamDecoder::new(kind, d, p, Basis::Z, commit, buffer, mode, rounds)?;
    let circuit = crate::memory::generate_repeat(kind, d, rounds, p, Basis::Z)?;
    stream_with(&dec, &circuit, correlated, batches, seed, threads)
}

/// Any circuit with a loop, sampled and window-decoded as it streams, its windows from a
/// template of its folded model (see `StreamDecoder::from_program`).
#[allow(clippy::too_many_arguments)]
pub fn stream_circuit(
    circuit: &crate::circuit::Circuit,
    commit: usize,
    buffer: usize,
    mode: crate::window::Mode,
    correlated: bool,
    batches: usize,
    seed: u64,
    threads: usize,
) -> Result<StreamOutcome, String> {
    let program = crate::dem_build::build(circuit, true, Some(1.0), true)?;
    let dec = crate::stream::StreamDecoder::from_program(&program, commit, buffer, mode, false)?;
    stream_with(&dec, circuit, correlated, batches, seed, threads)
}

fn stream_with(
    dec: &crate::stream::StreamDecoder,
    circuit: &crate::circuit::Circuit,
    correlated: bool,
    batches: usize,
    seed: u64,
    threads: usize,
) -> Result<StreamOutcome, String> {
    use crate::stream::run_stream;
    use crate::surface_code::Xorshift;
    use std::time::Instant;
    let plan = dec.plan()?;
    let sampler = crate::batch_sampler::BatchSampler::new(circuit)?;
    let threads = crate::parallel::resolve_threads(threads, batches);
    let start = Instant::now();
    // Per thread: failures, defects left unexplained, and window times.
    type Part = Result<(usize, usize, Vec<f64>), String>;
    let parts: Vec<Part> = std::thread::scope(|scope| {
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
                        let (pred, truth, u, ts) = run_stream(dec, plan, sampler, &mut rng, correlated, &mut scratches, Some(&clock))
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
    });
    let seconds = start.elapsed().as_secs_f64();
    let (mut failures, mut unexplained, mut times) = (0usize, 0usize, Vec::new());
    for part in parts {
        let (f, u, t) = part?;
        failures += f;
        unexplained += u;
        times.extend(t);
    }
    let windows = plan.specs.iter().zip(&plan.deps).map(|(s, deps)| (s.a, s.b, s.commit.0, s.commit.1, s.phase, deps.clone())).collect();
    Ok(StreamOutcome { failures, shots: batches * 64, unexplained, times, windows, seconds })
}

/// Union-find of b8 shots: the observables per shot, `None` where no correction exists.
pub fn union_find_shots(graph: &SparseGraph, packed: &[u8], nd: usize, num_shots: usize, threads: usize) -> Vec<Option<u64>> {
    let stride = nd.div_ceil(8);
    parallel(num_shots, threads, |range| {
        let mut scratch = graph.union_find_scratch();
        let mut defects = Vec::new();
        range
            .map(|s| {
                defects.clear();
                crate::shots::defects_from_b8(&packed[s * stride..(s + 1) * stride], nd, &mut defects);
                graph.decode_union_find(&mut scratch, &defects)
            })
            .collect()
    })
}

/// Windows for a model too long to unroll: a `StreamDecoder` from a template of its loop, its
/// plan, and each layer's detectors, so that shots given whole are fed in layer by layer.
pub struct StreamedWindows {
    dec: crate::stream::StreamDecoder,
    plan: crate::stream::StreamPlan,
    /// Per layer of the full model, its detectors in order.
    members: Vec<Vec<u32>>,
}

impl StreamedWindows {
    pub fn new(program: &crate::dem_program::DemProgram, commit: usize, buffer: usize, mode: crate::window::Mode) -> Result<StreamedWindows, String> {
        let dec = crate::stream::StreamDecoder::from_program(program, commit, buffer, mode, false)?;
        let plan = dec.plan()?;
        let times = program.detector_times()?;
        let mut distinct = times.clone();
        distinct.sort_by(|a, b| a.partial_cmp(b).expect("times are numbers"));
        distinct.dedup();
        if distinct.len() != plan.layers as usize {
            return Err(format!("the model has {} layers where its template's plan expects {}", distinct.len(), plan.layers));
        }
        let mut members = vec![Vec::new(); distinct.len()];
        for (d, t) in times.iter().enumerate() {
            let l = distinct.binary_search_by(|x| x.partial_cmp(t).expect("times are numbers")).expect("a layer of its own");
            members[l].push(d as u32);
        }
        // Every window's layers hold as many detectors as its template window's.
        let template = &dec.template.model.layers.members;
        for (i, spec) in plan.specs.iter().enumerate() {
            let t = dec.template.specs[plan.template[i]];
            for k in 0..(spec.b - spec.a) {
                if members[(spec.a + k) as usize].len() != template[(t.a + k) as usize].len() {
                    return Err(format!("window {i}'s layer {} does not match its template's", spec.a + k));
                }
            }
        }
        Ok(StreamedWindows { dec, plan, members })
    }

    pub fn windows(&self) -> Vec<WindowInfo> {
        self.plan
            .specs
            .iter()
            .zip(&self.plan.deps)
            .map(|(s, deps)| (s.a, s.b, s.commit.0, s.commit.1, s.phase, deps.clone()))
            .collect()
    }
}

/// One shot's window decoding: its observables, the defects left unexplained, each window's
/// decode seconds, and whether a window refused it.
pub type WindowShot = (u64, usize, Vec<f64>, bool);

/// Streamed window decoding of b8 shots, 64 at a time, in the shape `window_shots` gives:
/// (observables, defects left unexplained, each window's decode seconds, refused). Timings are
/// the first shot's of each 64 (the rest NaN): the stream decodes 64 at once.
pub fn streamed_window_shots(
    sw: &StreamedWindows,
    packed: &[u8],
    nd: usize,
    num_shots: usize,
    correlated: bool,
    threads: usize,
    timings: bool,
) -> Vec<WindowShot> {
    use crate::stream::Stream;
    let stride = nd.div_ceil(8);
    let nw = sw.plan.specs.len();
    let batches = num_shots.div_ceil(64);
    let per_batch: Vec<Vec<WindowShot>> = parallel(batches, threads, |range| {
        let mut scratches = sw.dec.scratches();
        let origin = std::time::Instant::now();
        let clock = move || origin.elapsed().as_secs_f64();
        range
            .map(|b| {
                let lanes = (num_shots - b * 64).min(64);
                let mut stream = Stream::new(&sw.dec, &sw.plan, correlated, &mut scratches, timings.then_some(&clock as &dyn Fn() -> f64));
                let mut words = Vec::new();
                let mut failed = false;
                for layer in &sw.members {
                    words.clear();
                    words.extend(layer.iter().map(|&d| {
                        let (byte, bit) = (d as usize / 8, d % 8);
                        (0..lanes).fold(0u64, |w, lane| w | (u64::from((packed[(b * 64 + lane) * stride + byte] >> bit) & 1) << lane))
                    }));
                    if stream.push_layer(&words).is_err() {
                        failed = true;
                        break;
                    }
                }
                let finished = if failed { None } else { stream.finish().ok() };
                match finished {
                    Some((stream, unexplained)) => (0..lanes)
                        .map(|lane| {
                            let times = if timings && lane == 0 { stream.times.clone() } else if timings { vec![f64::NAN; nw] } else { Vec::new() };
                            (stream.predictions[lane], if lane == 0 { unexplained } else { 0 }, times, false)
                        })
                        .collect(),
                    None => (0..lanes).map(|_| (0, 0, if timings { vec![f64::NAN; nw] } else { Vec::new() }, true)).collect(),
                }
            })
            .collect()
    });
    per_batch.into_iter().flatten().collect()
}

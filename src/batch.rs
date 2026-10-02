//! The loops over shots that the Python bindings and the crate's API share: each decoder over
//! Stim's b8 rows, cut across threads, results in shot order.

use crate::belief::BeliefMatching;
use crate::osd::BpOsd;
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

/// BP+OSD of b8 shots on a model's faults: (observables, BP converged).
pub fn bposd_shots(dec: &BpOsd, obs: &[u64], packed: &[u8], nd: usize, num_shots: usize, threads: usize) -> Vec<(u64, u8)> {
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
                let o = dec.decode(&syndrome, &mut work);
                let pred = work.correction.iter().zip(obs).filter(|(c, _)| **c != 0).fold(0u64, |a, (_, o)| a ^ o);
                (pred, u8::from(o.converged))
            })
            .collect()
    })
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
) -> Vec<(u64, usize, Vec<f64>, bool)> {
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


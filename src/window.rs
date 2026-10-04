//! Decoding a stream of rounds window by window, as a real-time decoder must.
//!
//! WHY THIS EXISTS
//! ---------------
//! A global decoder waits for the whole experiment and matches everything at
//! once. A quantum computer cannot wait: Willow runs a round every 1.1 µs, and
//! a logical measurement's outcome is needed while the rounds keep coming. A
//! window decoder matches a few rounds at a time. It decodes a window of
//! `commit + buffer` rounds, keeps (commits) only the corrections touching the
//! first `commit` rounds, and carries the rest forward: the buffer exists so
//! that a defect near the window's end can wait for a partner not yet seen,
//! through a boundary standing in for the future.
//!
//! Two schedules:
//! - **Sliding** windows run one after another; each needs the last one's
//!   commits, so a stream uses one core.
//! - **Parallel** windows (Skoric et al., arXiv:2209.08552; Tan et al.,
//!   arXiv:2209.09219) come in two layers. Layer A's windows are spaced apart,
//!   with buffers on both sides and boundaries standing in for both past and
//!   future, and decode independently, on as many cores as there are. Layer
//!   B's windows then fill the gaps between A's commit regions, with A's
//!   commits already in place, again independently. Throughput grows with the
//!   number of cores.
//!
//! Windows are cut from the model by layer: a detector's layer is its time
//! coordinate's index among the model's distinct times. An edge from inside a
//! window to a layer beyond it becomes a boundary half-edge of its own — kept
//! apart from the real boundary edge, since it stands for a different thing.
//!
//! Committing: an edge of a window's correction is committed when either end
//! lies in the commit region. Its observables go into the prediction, and both
//! its ends' defects are toggled, so a correction reaching out of the region
//! leaves a defect for the region it reaches. When the stream is done, no
//! defect may be left unexplained; that is checked on every shot.

use crate::dem::Dem;
use crate::dem_decoder::{merged_edges, DecodeError};
use crate::sparse::{Correlations, Scratch, SparseGraph};

const NONE: u32 = u32::MAX;

/// Every detector's layer, and the detectors of each layer.
pub struct Layers {
    pub of: Vec<u32>,
    pub members: Vec<Vec<u32>>,
}

impl Layers {
    pub fn from_dem(dem: &Dem) -> Result<Layers, String> {
        let mut times = Vec::with_capacity(dem.num_detectors);
        for d in 0..dem.num_detectors {
            let t = *dem
                .detector_coords
                .get(d)
                .and_then(|c| c.last())
                .ok_or_else(|| {
                    format!("detector D{d} has no time coordinate; windows need them")
                })?;
            times.push(t);
        }
        let mut distinct = times.clone();
        distinct.sort_by(|a, b| a.partial_cmp(b).expect("times are numbers"));
        distinct.dedup();
        let of: Vec<u32> = times
            .iter()
            .map(|t| {
                distinct
                    .binary_search_by(|x| x.partial_cmp(t).unwrap())
                    .unwrap() as u32
            })
            .collect();
        let mut members = vec![Vec::new(); distinct.len()];
        for (d, &l) in of.iter().enumerate() {
            members[l as usize].push(d as u32);
        }
        Ok(Layers { of, members })
    }

    pub fn count(&self) -> usize {
        self.members.len()
    }
}

/// A detector error model prepared for windows: its merged edges, its layers,
/// and its correlated-matching rules.
pub struct Model {
    pub num_detectors: usize,
    pub edges: Vec<(u32, u32, f64, u64)>,
    pub layers: Layers,
    corr: Correlations,
}

/// One window's graph, with the way back to the model's detectors and edges.
pub struct Window {
    pub layers: (u32, u32),
    pub graph: SparseGraph,
    pub corr: Correlations,
    /// Local node → the model's detector; sorted, so a detector's local index
    /// is found by binary search.
    pub global_node: Vec<u32>,
    /// Local edge → the model's edge.
    pub global_edge: Vec<u32>,
}

impl Model {
    pub fn new(dem: &Dem) -> Result<Model, String> {
        let (edges, _) = merged_edges(dem)?;
        let layers = Layers::from_dem(dem)?;
        let full = SparseGraph::from_edges(dem.num_detectors, &edges);
        let corr = Correlations::from_dem(dem, &full)?;
        Ok(Model {
            num_detectors: dem.num_detectors,
            edges,
            layers,
            corr,
        })
    }

    /// The window over layers `[a, b)`. With `past` (or `future`), an edge to
    /// a layer before `a` (or from `b` on) becomes a boundary half-edge of its
    /// end inside; without, it is dropped.
    pub fn window(&self, a: u32, b: u32, past: bool, future: bool) -> Window {
        let mut nodes: Vec<u32> = (a..b)
            .flat_map(|l| self.layers.members[l as usize].iter().copied())
            .collect();
        nodes.sort_unstable();
        let n = nodes.len() as u32;
        let boundary = self.num_detectors as u32;
        let local = |g: u32| nodes.binary_search(&g).ok().map(|i| i as u32);
        let outside_kept = |g: u32| {
            let l = self.layers.of[g as usize];
            (l >= b && future) || (l < a && past)
        };
        let mut list: Vec<(u32, u32, f64, u64)> = Vec::new();
        let mut global_edge = Vec::new();
        for (gid, &(u, v, p, o)) in self.edges.iter().enumerate() {
            let lu = local(u);
            let kept = if v == boundary {
                lu.map(|x| (x, n))
            } else {
                match (lu, local(v)) {
                    (Some(x), Some(y)) => Some((x.min(y), x.max(y))),
                    (Some(x), None) if outside_kept(v) => Some((x, n)),
                    (None, Some(y)) if outside_kept(u) => Some((y, n)),
                    _ => None,
                }
            };
            if let Some((x, y)) = kept {
                list.push((x, y, p, o));
                global_edge.push(gid as u32);
            }
        }
        let graph = SparseGraph::from_edges(n as usize, &list);
        let mut map = vec![NONE; self.edges.len()];
        for (i, &g) in global_edge.iter().enumerate() {
            map[g as usize] = i as u32;
        }
        let corr = self.corr.restrict(&map, list.len());
        Window {
            layers: (a, b),
            graph,
            corr,
            global_node: nodes,
            global_edge,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Sliding,
    Parallel,
}

/// Where one window sits: layers `[a, b)`, its commit region, which virtual
/// boundaries it has, and its phase (windows of one phase are independent;
/// phases run in order).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub a: u32,
    pub b: u32,
    pub commit: (u32, u32),
    pub past: bool,
    pub future: bool,
    pub phase: usize,
}

/// A schedule's shape is sound: a window commits at least one layer, and
/// parallel windows have a buffer. Layer B lives in the gaps between A's
/// commit regions, 2B layers wide; with no buffer there are no gaps, and a
/// correction an A window reaches back into its neighbour with would have
/// nowhere to be resolved.
pub fn check_schedule(commit: usize, buffer: usize, mode: Mode) -> Result<(), String> {
    if commit == 0 {
        return Err("a window must commit at least one layer".into());
    }
    if mode == Mode::Parallel && buffer == 0 {
        return Err("parallel windows need a buffer of at least one layer".into());
    }
    let span = buffer.checked_mul(2).and_then(|b| b.checked_add(commit));
    if span.is_none_or(|s| s > u32::MAX as usize) {
        return Err(format!("commit {commit} and buffer {buffer} are too large"));
    }
    Ok(())
}

/// The windows of a schedule over `total` layers, in the order they are
/// planned (sliding: in time; parallel: layer A in time, then layer B).
pub fn plan(total: u32, commit: usize, buffer: usize, mode: Mode) -> Result<Vec<Spec>, String> {
    check_schedule(commit, buffer, mode)?;
    let (c, b) = (commit as u32, buffer as u32);
    let mut out = Vec::new();
    match mode {
        Mode::Sliding => {
            let mut s = 0u32;
            while s < total {
                let last = s + c + b >= total;
                let end = if last { total } else { s + c + b };
                let commit = if last { (s, total) } else { (s, s + c) };
                out.push(Spec {
                    a: s,
                    b: end,
                    commit,
                    past: false,
                    future: !last,
                    phase: out.len(),
                });
                s = commit.1;
            }
        }
        Mode::Parallel => {
            let mut starts = Vec::new();
            let mut s = 0u32;
            while s < total {
                let (a, e) = (s.saturating_sub(b), (s + c + b).min(total));
                out.push(Spec {
                    a,
                    b: e,
                    commit: (s, (s + c).min(total)),
                    past: a > 0,
                    future: e < total,
                    phase: 0,
                });
                starts.push(s);
                s += c + 2 * b;
            }
            for (i, &s) in starts.iter().enumerate() {
                let gap = (
                    (s + c).min(total),
                    starts.get(i + 1).copied().unwrap_or(total),
                );
                if gap.0 < gap.1 {
                    out.push(Spec {
                        a: gap.0,
                        b: gap.1,
                        commit: gap,
                        past: false,
                        future: false,
                        phase: 1,
                    });
                }
            }
        }
    }
    Ok(out)
}

/// A window's graph and its commit region.
pub struct Planned {
    pub window: Window,
    pub commit: (u32, u32),
}

/// A model cut into windows for one schedule.
pub struct WindowDecoder {
    pub model: Model,
    pub specs: Vec<Spec>,
    pub windows: Vec<Planned>,
    /// Windows in each phase: within a phase they are independent; phases run
    /// in order. Sliding has one window per phase.
    pub phases: Vec<Vec<usize>>,
    pub mode: Mode,
}

/// A decoded shot.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    pub observables: u64,
    /// Defects left once every window has committed; zero on a correct decode.
    pub unexplained: usize,
}

impl WindowDecoder {
    pub fn new(
        model: Model,
        commit: usize,
        buffer: usize,
        mode: Mode,
    ) -> Result<WindowDecoder, String> {
        let specs = plan(model.layers.count() as u32, commit, buffer, mode)?;
        let windows = specs
            .iter()
            .map(|s| Planned {
                window: model.window(s.a, s.b, s.past, s.future),
                commit: s.commit,
            })
            .collect();
        let mut phases: Vec<Vec<usize>> =
            vec![Vec::new(); specs.iter().map(|s| s.phase + 1).max().unwrap_or(0)];
        for (i, s) in specs.iter().enumerate() {
            phases[s.phase].push(i);
        }
        Ok(WindowDecoder {
            model,
            specs,
            windows,
            phases,
            mode,
        })
    }

    /// One scratch per window, sized for its graph.
    pub fn scratches(&self) -> Vec<Scratch> {
        self.windows
            .iter()
            .map(|w| Scratch::new(&w.window.graph))
            .collect()
    }

    /// Decode window `wi` given the live defects so far, commit its
    /// correction, and update `live` and `observables` accordingly.
    pub fn decode_window(
        &self,
        wi: usize,
        live: &mut [bool],
        observables: &mut u64,
        correlated: bool,
        scratch: &mut Scratch,
    ) -> Result<(), DecodeError> {
        let w = &self.windows[wi];
        let defects: Vec<u32> = w
            .window
            .global_node
            .iter()
            .enumerate()
            .filter(|(_, &g)| live[g as usize])
            .map(|(i, _)| i as u32)
            .collect();
        let edges = if correlated {
            w.window
                .graph
                .decode_correlated_edge_ids(&w.window.corr, scratch, &defects)?
        } else {
            w.window.graph.decode_edge_ids(scratch, &defects)?
        };
        self.commit(wi, &edges, live, observables);
        Ok(())
    }

    fn commit(&self, wi: usize, edges: &[u32], live: &mut [bool], observables: &mut u64) {
        let w = &self.windows[wi];
        let boundary = self.model.num_detectors as u32;
        let (c0, c1) = w.commit;
        let inside = |node: u32| {
            node != boundary && {
                let l = self.model.layers.of[node as usize];
                l >= c0 && l < c1
            }
        };
        for &le in edges {
            let (u, v, _, o) = self.model.edges[w.window.global_edge[le as usize] as usize];
            if inside(u) || inside(v) {
                *observables ^= o;
                live[u as usize] ^= true;
                if v != boundary {
                    live[v as usize] ^= true;
                }
            }
        }
    }

    /// Decode one shot, every window in its schedule's order.
    pub fn decode_with(
        &self,
        defects: &[u32],
        correlated: bool,
        scratches: &mut [Scratch],
    ) -> Result<Outcome, DecodeError> {
        let mut live = vec![false; self.model.num_detectors];
        for &d in defects {
            live[d as usize] ^= true;
        }
        let mut observables = 0u64;
        for phase in &self.phases {
            for &wi in phase {
                self.decode_window(
                    wi,
                    &mut live,
                    &mut observables,
                    correlated,
                    &mut scratches[wi],
                )?;
            }
        }
        Ok(Outcome {
            observables,
            unexplained: live.iter().filter(|&&b| b).count(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Basis;
    use crate::dem_decoder::DemDecoder;
    use crate::frame_sampler::FrameSampler;
    use crate::memory::{generate, CodeKind, NoiseModel};
    use crate::surface_code::Xorshift;

    fn defects_of(dets: &[bool]) -> Vec<u32> {
        dets.iter()
            .enumerate()
            .filter(|x| *x.1)
            .map(|x| x.0 as u32)
            .collect()
    }

    fn sd6(kind: CodeKind, d: usize, rounds: usize, p: f64) -> (crate::circuit::Circuit, Dem) {
        let c = generate(kind, d, rounds, NoiseModel::Sd6 { p }, Basis::Z).unwrap();
        let dem = Dem::from_circuit(&c).unwrap();
        (c, dem)
    }

    #[test]
    fn unsound_schedules_are_refused() {
        let (_, dem) = sd6(CodeKind::Rotated, 3, 5, 0.005);
        let refused = |c, b, mode| {
            WindowDecoder::new(Model::new(&dem).unwrap(), c, b, mode)
                .err()
                .unwrap_or_default()
        };
        assert!(refused(2, 0, Mode::Parallel).contains("buffer of at least one layer"));
        assert!(refused(0, 2, Mode::Parallel).contains("at least one layer"));
        assert!(refused(0, 2, Mode::Sliding).contains("at least one layer"));
        assert!(WindowDecoder::new(Model::new(&dem).unwrap(), 2, 0, Mode::Sliding).is_ok());
    }

    #[test]
    fn layers_follow_the_time_coordinate() {
        let dem = Dem::parse("detector(0, 0, 0) D0\ndetector(1, 0, 0) D1\ndetector(0, 0, 2) D2\ndetector(0, 0, 1) D3\nerror(0.1) D0 D1\n").unwrap();
        let l = Layers::from_dem(&dem).unwrap();
        assert_eq!(l.of, vec![0, 0, 2, 1]);
        assert_eq!(l.members, vec![vec![0, 1], vec![3], vec![2]]);
        assert!(Layers::from_dem(&Dem::parse("error(0.1) D0").unwrap()).is_err());
    }

    #[test]
    fn a_window_over_everything_is_the_whole_graph() {
        let (_, dem) = sd6(CodeKind::Rotated, 3, 3, 0.005);
        let model = Model::new(&dem).unwrap();
        let full = DemDecoder::new(&dem).unwrap();
        let w = model.window(0, model.layers.count() as u32, false, false);
        let (g, h) = (&w.graph, full.graph());
        assert_eq!(
            w.global_node,
            (0..dem.num_detectors as u32).collect::<Vec<_>>()
        );
        assert_eq!(
            (g.num_edges(), g.to.clone(), g.w.clone(), g.obs.clone()),
            (h.num_edges(), h.to.clone(), h.w.clone(), h.obs.clone())
        );
        for e in 0..g.num_edges() as u32 {
            assert_eq!(w.corr.rules_of(e), full.correlations().rules_of(e));
        }
    }

    #[test]
    fn edges_leaving_a_window_become_its_own_boundaries() {
        let dem = Dem::parse(
            "detector(0, 0, 0) D0\ndetector(0, 0, 1) D1\ndetector(0, 0, 2) D2\nerror(0.1) D0 D1\nerror(0.2) D1 D2\nerror(0.3) D1\n",
        )
        .unwrap();
        let model = Model::new(&dem).unwrap();
        let both = model.window(1, 2, true, true);
        assert_eq!(both.global_node, vec![1]);
        let targets: Vec<u32> = both.graph.edges(0).map(|e| both.graph.to[e]).collect();
        assert_eq!(
            targets.len(),
            3,
            "past, real and future boundaries, kept apart"
        );
        let mut from: Vec<u32> = both.global_edge.clone();
        from.sort();
        assert_eq!(from, vec![0, 1, 2]);
        assert_eq!(
            model.window(1, 2, false, false).graph.num_edges(),
            1,
            "only the real boundary"
        );
        assert_eq!(model.window(1, 2, false, true).graph.num_edges(), 2);
    }

    #[test]
    fn edge_ids_are_the_traced_edges() {
        let (c, dem) = sd6(CodeKind::Rotated, 3, 3, 0.006);
        let dec = DemDecoder::new(&dem).unwrap();
        let sampler = FrameSampler::new(&c).unwrap();
        let mut rng = Xorshift::new(2);
        let mut s = Scratch::new(dec.graph());
        for _ in 0..200 {
            let defects = defects_of(&sampler.sample(&mut rng).detectors);
            let mut ids = dec.graph().decode_edge_ids(&mut s, &defects).unwrap();
            let mut ends: Vec<u32> = dec
                .decode_to_edges(&defects)
                .unwrap()
                .iter()
                .map(|&(u, v)| dec.graph().edge_id(u, v).unwrap())
                .collect();
            ids.sort();
            ends.sort();
            assert_eq!(ids, ends);
        }
    }

    /// Pass two's edges: a correction of the shot, whose weight in the
    /// lowered weights is pass two's optimum.
    #[test]
    fn pass_two_edges_are_its_optimal_correction() {
        let (c, dem) = sd6(CodeKind::Xzzx, 5, 5, 0.006);
        let dec = DemDecoder::new(&dem).unwrap();
        let (g, corr) = (dec.graph(), dec.correlations());
        let sampler = FrameSampler::new(&c).unwrap();
        let mut rng = Xorshift::new(8);
        let mut s = Scratch::new(g);
        for _ in 0..200 {
            let defects = defects_of(&sampler.sample(&mut rng).detectors);
            let first = g.decode_edge_ids(&mut s, &defects).unwrap();
            let second = g
                .decode_correlated_edge_ids(corr, &mut s, &defects)
                .unwrap();
            // The lowered weights pass one's edges imply.
            let mut w: Vec<i64> = (0..g.num_edges() as u32).map(|e| g.weight_of(e)).collect();
            for &e in &first {
                for (a, _, iw) in corr.rules_of(e) {
                    w[a as usize] = w[a as usize].min(iw);
                }
            }
            let mut syndrome = vec![false; g.num_nodes];
            for &e in &second {
                let (u, v) = g.edge_ends(e);
                syndrome[u as usize] ^= true;
                if (v as usize) < g.num_nodes {
                    syndrome[v as usize] ^= true;
                }
            }
            assert_eq!(defects_of(&syndrome), defects);
            let weight: i64 = second.iter().map(|&e| w[e as usize]).sum();
            assert_eq!(
                weight,
                g.decode_correlated(corr, &mut s, &defects).unwrap().iweight
            );
        }
    }

    /// With a commit region as long as the stream there is one window, the
    /// whole graph, so its correction is exactly the global decoder's traced
    /// correction, plain and correlated.
    #[test]
    fn one_window_is_exactly_global_decoding() {
        let mut rng = Xorshift::new(5);
        for kind in [CodeKind::Rotated, CodeKind::Xzzx] {
            for d in [3usize, 5] {
                let (c, dem) = sd6(kind, d, d, 0.006);
                let dec = DemDecoder::new(&dem).unwrap();
                let sampler = FrameSampler::new(&c).unwrap();
                for mode in [Mode::Sliding, Mode::Parallel] {
                    let wd = WindowDecoder::new(Model::new(&dem).unwrap(), 1000, 3, mode).unwrap();
                    assert_eq!(wd.windows.len(), 1);
                    let mut scratches = wd.scratches();
                    let mut s = Scratch::new(dec.graph());
                    for _ in 0..100 {
                        let defects = defects_of(&sampler.sample(&mut rng).detectors);
                        for correlated in [false, true] {
                            let got = wd
                                .decode_with(&defects, correlated, &mut scratches)
                                .unwrap();
                            let ids = if correlated {
                                dec.graph()
                                    .decode_correlated_edge_ids(
                                        dec.correlations(),
                                        &mut s,
                                        &defects,
                                    )
                                    .unwrap()
                            } else {
                                dec.graph().decode_edge_ids(&mut s, &defects).unwrap()
                            };
                            let want = ids.iter().fold(0u64, |o, &e| {
                                o ^ dec.graph().obs[dec.graph().halves[e as usize][0] as usize]
                            });
                            assert_eq!(
                                got,
                                Outcome {
                                    observables: want,
                                    unexplained: 0
                                },
                                "{kind:?} d = {d} {mode:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Every shot's defects are explained by the committed edges, whatever
    /// the window sizes: nothing is left over and nothing is committed twice.
    #[test]
    fn every_defect_is_explained() {
        let mut rng = Xorshift::new(6);
        for d in [3usize, 5] {
            let (c, dem) = sd6(CodeKind::Rotated, d, 10, 0.008);
            let sampler = FrameSampler::new(&c).unwrap();
            let shots: Vec<Vec<u32>> = (0..40)
                .map(|_| defects_of(&sampler.sample(&mut rng).detectors))
                .collect();
            for mode in [Mode::Sliding, Mode::Parallel] {
                for commit in 1..=3 {
                    for buffer in usize::from(mode == Mode::Parallel)..=3 {
                        let wd =
                            WindowDecoder::new(Model::new(&dem).unwrap(), commit, buffer, mode)
                                .unwrap();
                        let mut scratches = wd.scratches();
                        for defects in &shots {
                            for correlated in [false, true] {
                                let got =
                                    wd.decode_with(defects, correlated, &mut scratches).unwrap();
                                assert_eq!(
                                    got.unexplained, 0,
                                    "d = {d} {mode:?} C = {commit} B = {buffer}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// With B = C = d, windowed decoding fails about as often as global
    /// decoding: the difference in failures is within 4σ of the shots on which
    /// the two differ.
    #[test]
    fn windows_of_size_d_decode_as_well_as_global() {
        let (c, dem) = sd6(CodeKind::Rotated, 5, 30, 0.005);
        let dec = DemDecoder::new(&dem).unwrap();
        let sampler = FrameSampler::new(&c).unwrap();
        let mut rng = Xorshift::new(99);
        let shots: Vec<(Vec<u32>, u64)> = (0..3000)
            .map(|_| {
                let s = sampler.sample(&mut rng);
                (defects_of(&s.detectors), s.observables)
            })
            .collect();
        for mode in [Mode::Sliding, Mode::Parallel] {
            let wd = WindowDecoder::new(Model::new(&dem).unwrap(), 5, 5, mode).unwrap();
            let mut scratches = wd.scratches();
            for correlated in [false, true] {
                let (mut global, mut windowed, mut differ) = (0i64, 0i64, 0i64);
                for (defects, obs) in &shots {
                    let g = if correlated {
                        dec.decode_correlated(defects)
                    } else {
                        dec.decode(defects)
                    }
                    .unwrap()
                    .observables;
                    let w = wd.decode_with(defects, correlated, &mut scratches).unwrap();
                    assert_eq!(w.unexplained, 0);
                    global += ((g ^ obs) & 1) as i64;
                    windowed += ((w.observables ^ obs) & 1) as i64;
                    differ += ((g ^ w.observables) & 1) as i64;
                }
                println!("{mode:?} correlated {correlated}: global {global}, windowed {windowed}, differ {differ} of 3000");
                assert!(
                    (windowed - global).abs() as f64 <= 4.0 * (differ.max(1) as f64).sqrt() + 1.0,
                    "{mode:?} {correlated}: {windowed} vs {global}"
                );
            }
        }
    }
}

//! Decoding a stream too long to model: a million rounds of a memory.
//!
//! WHY THIS EXISTS
//! ---------------
//! Google decoded a distance-5 memory in real time over a million rounds. The
//! window decoders in `window` cut their windows from the experiment's error
//! model, and a million-round model does not fit in memory and would take
//! hours to build. It is not needed: a memory circuit repeats, so every window
//! away from the stream's two ends sees the same graph. This decoder builds the
//! error model of a short circuit of the same code once — long enough that its
//! middle windows are away from both of its ends — and decodes each window of
//! the stream with the graph of the window in the same position: the first few
//! and last few from the short model's own ends, the rest from its middle.
//! A stream window and its template must have the same shape once shifted (the
//! decoder checks every one), and on a 60-round stream the whole scheme agrees
//! shot for shot with windows cut from the full 60-round model (the tests).
//!
//! Rounds are pushed in as the sampler makes them; a window is decoded as soon
//! as its last round has arrived and the windows it depends on are done; and a
//! round is let go once no window still needs it. 64 streams run at once, one
//! per bit of each detection word.

use std::collections::VecDeque;

use crate::circuit::Basis;
use crate::dem_decoder::DecodeError;
use crate::dem_program::{DemInstr, DemProgram};
use crate::memory::CodeKind;
use crate::sparse::Scratch;
use crate::window::{check_schedule, plan, Layers, Mode, Model, Spec, WindowDecoder};

/// Windows matched from each end of the stream to the template's ends, at least.
const EDGE: usize = 3;

pub struct StreamDecoder {
    pub template: WindowDecoder,
    /// The template's layers, and the stream's.
    pub template_layers: u32,
    pub stream_layers: u32,
    /// Windows matched from each end of the stream to the template's ends.
    edge: usize,
    /// Windows in one cycle of the middle: a whole number of window periods spanning a whole
    /// number of the loop's passes.
    cycle: usize,
    /// Per template detector: its layer and its rank within the layer.
    place: Vec<(u32, u32)>,
    /// Per template window, per layer from its first, per rank: its local node.
    local: Vec<Vec<Vec<u32>>>,
    commit: usize,
    buffer: usize,
    mode: Mode,
}

/// A stream's windows, each tied to a template window and an offset in layers.
pub struct StreamPlan {
    pub layers: u32,
    pub specs: Vec<Spec>,
    pub template: Vec<usize>,
    pub offset: Vec<u32>,
    /// The order windows are decoded in, as their rounds arrive.
    pub order: Vec<usize>,
    /// Per window: the layer that must have arrived, and the windows it waits for.
    pub ready_layer: Vec<u32>,
    pub deps: Vec<Vec<usize>>,
    /// The smallest first layer among windows `order[i..]`, for letting rounds go.
    suffix_min_a: Vec<u32>,
}

fn by_time(specs: &[Spec]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..specs.len()).collect();
    idx.sort_by_key(|&i| specs[i].commit.0);
    idx
}

/// The distinct time coordinates of the detectors a model declares.
fn declared_layers(program: &DemProgram) -> Result<usize, String> {
    let mut times: Vec<f64> = Vec::new();
    for ins in program.flattened()?.instrs {
        if let DemInstr::Detector { coords, .. } = ins {
            times.push(
                *coords
                    .last()
                    .ok_or("a detector has no time coordinate; windows need them")?,
            );
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).expect("times are numbers"));
    times.dedup();
    Ok(times.len())
}

/// Parallel layer-B windows wait for the layer-A windows bordering their
/// commit region; sliding windows wait for the one before.
pub fn dependencies(specs: &[Spec], mode: Mode) -> Vec<Vec<usize>> {
    let t = by_time(specs);
    let mut deps = vec![Vec::new(); specs.len()];
    for (j, &i) in t.iter().enumerate() {
        deps[i] = match mode {
            Mode::Sliding => t.get(j.wrapping_sub(1)).copied().into_iter().collect(),
            Mode::Parallel if specs[i].phase == 1 => (0..specs.len())
                .filter(|&k| {
                    specs[k].phase == 0
                        && (specs[k].commit.1 == specs[i].commit.0
                            || specs[k].commit.0 == specs[i].commit.1)
                })
                .collect(),
            Mode::Parallel => Vec::new(),
        };
    }
    deps
}

impl StreamDecoder {
    /// A decoder for a rotated or XZZX SD6 memory of `stream_rounds` rounds.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kind: CodeKind,
        d: usize,
        p: f64,
        basis: Basis,
        commit: usize,
        buffer: usize,
        mode: Mode,
        stream_rounds: usize,
    ) -> Result<StreamDecoder, String> {
        let circuit = crate::memory::generate_repeat(kind, d, stream_rounds, p, basis)?;
        let program = crate::dem_build::build(&circuit, true, None, true)?;
        StreamDecoder::from_program(&program, commit, buffer, mode, true)
    }

    /// A decoder for the model `program` describes, however long, built from a template of it:
    /// the same model with its longest top-level `repeat` run fewer times, long enough that its
    /// middle windows sit inside the repetition and its end windows cover what lies before and
    /// after it. Windows of the stream away from its ends are the template's middle windows
    /// shifted by whole passes of the loop; the rest are the template's end windows. `merged`
    /// unrolls each model as `to_dem_merged` does, else as `to_dem`. Refuses (and a caller may
    /// cut windows from the whole model instead) a model with no loop, one whose passes add a
    /// varying number of layers, or a schedule whose period is not a whole number of passes.
    pub fn from_program(
        program: &DemProgram,
        commit: usize,
        buffer: usize,
        mode: Mode,
        merged: bool,
    ) -> Result<StreamDecoder, String> {
        check_schedule(commit, buffer, mode)?;
        let period = match mode {
            Mode::Sliding => commit,
            Mode::Parallel => commit + 2 * buffer,
        };
        let windows_per_period = if mode == Mode::Sliding { 1 } else { 2 };
        let flat = |p: &DemProgram| {
            if merged {
                p.to_dem_merged()
            } else {
                p.to_dem()
            }
        };
        let layer_count =
            |p: &DemProgram| -> Result<usize, String> { Ok(Layers::from_dem(&flat(p)?)?.count()) };
        let (index, count) = program
            .instrs
            .iter()
            .enumerate()
            .filter_map(|(i, ins)| match ins {
                DemInstr::Repeat { count, .. } => Some((i, *count)),
                _ => None,
            })
            .max_by_key(|&(_, c)| c)
            .ok_or("the model has no top-level loop to take a template from")?;
        let with_count = |c: u64| {
            let mut q = program.clone();
            if let DemInstr::Repeat { count, .. } = &mut q.instrs[index] {
                *count = c;
            }
            q
        };
        // Layers per pass, the same from pass to pass.
        let (l2, l3, l4) = (
            layer_count(&with_count(2))?,
            layer_count(&with_count(3))?,
            layer_count(&with_count(4))?,
        );
        let per = l3
            .checked_sub(l2)
            .filter(|&x| x > 0 && l4 == l3 + x)
            .ok_or("the loop's passes do not each add the same number of layers")?;
        // The middle repeats every lcm(period, per) layers: `periods` window periods.
        let gcd = |mut a: usize, mut b: usize| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        };
        let periods = per / gcd(period, per);
        let cycle_layers = period * periods;
        // Layers declared before the loop and after it. (A model's part before the loop alone
        // has faults on detectors declared later, so its layers are counted from its own
        // declarations.)
        let before = declared_layers(&DemProgram {
            instrs: program.instrs[..index].to_vec(),
        })?;
        let after = layer_count(&with_count(0))?
            .checked_sub(before)
            .ok_or("the model's layers before its loop outnumber the model's")?;
        let stride = period / windows_per_period;
        // End windows reach past the layers outside the loop and a buffer beyond them; every
        // window after them sits wholly inside the repetition.
        let edge = EDGE.max((before.max(after) + 1 + buffer).div_ceil(stride.max(1)) + 1);
        let full = l2 as u64 + (count.saturating_sub(2)) * per as u64;
        let min_layers =
            ((2 * edge / windows_per_period + 1 + periods) * period + buffer + 2) as u64;
        // Passes for the template: enough layers, and the stream's extra layers a whole number
        // of window periods.
        let passes = if count <= 2 || full <= min_layers + cycle_layers as u64 {
            count
        } else {
            (2..count)
                .find(|&c| {
                    let layers = l2 as u64 + (c - 2) * per as u64;
                    layers >= min_layers
                        && ((count - c) * per as u64).is_multiple_of(cycle_layers as u64)
                })
                .unwrap_or(count)
        };
        let template_program = with_count(passes);
        let dem = flat(&template_program)?;
        let template = WindowDecoder::new(Model::new(&dem)?, commit, buffer, mode)?;
        let template_layers = template.model.layers.count() as u32;
        let stream_layers = u32::try_from(template_layers as u64 + (count - passes) * per as u64)
            .map_err(|_| "too many layers")?;
        let layers = &template.model.layers;
        let mut place = vec![(0u32, 0u32); template.model.num_detectors];
        for (l, members) in layers.members.iter().enumerate() {
            for (r, &det) in members.iter().enumerate() {
                place[det as usize] = (l as u32, r as u32);
            }
        }
        let local = template
            .windows
            .iter()
            .map(|w| {
                (w.window.layers.0..w.window.layers.1)
                    .map(|l| {
                        layers.members[l as usize]
                            .iter()
                            .map(|det| {
                                w.window
                                    .global_node
                                    .binary_search(det)
                                    .expect("the window holds its layers")
                                    as u32
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let cycle = windows_per_period * periods;
        Ok(StreamDecoder {
            template,
            template_layers,
            stream_layers,
            edge,
            cycle,
            place,
            local,
            commit,
            buffer,
            mode,
        })
    }

    /// The stream's windows, each tied to its template window.
    pub fn plan(&self) -> Result<StreamPlan, String> {
        let edge = self.edge;
        let layers = self.stream_layers;
        let specs = plan(layers, self.commit, self.buffer, self.mode)?;
        let (t_specs, s_time, t_time) = (
            &self.template.specs,
            by_time(&specs),
            by_time(&self.template.specs),
        );
        let (n_s, n_t) = (specs.len(), t_specs.len());
        let per = self.cycle;
        let (mut template, mut offset) = (vec![0usize; n_s], vec![0u32; n_s]);
        for (j, &i) in s_time.iter().enumerate() {
            let tj = if n_s == n_t || j < edge {
                j
            } else if j >= n_s - edge {
                n_t - (n_s - j)
            } else {
                edge + (j - edge) % per
            };
            let (s, t) = (specs[i], t_specs[t_time[tj]]);
            let off =
                s.a.checked_sub(t.a)
                    .ok_or("a stream window sits before its template")?;
            let shifted = (
                t.a + off,
                t.b + off,
                t.commit.0 + off,
                t.commit.1 + off,
                t.past,
                t.future,
                t.phase.min(1),
            );
            if shifted
                != (
                    s.a,
                    s.b,
                    s.commit.0,
                    s.commit.1,
                    s.past,
                    s.future,
                    s.phase.min(1),
                )
            {
                return Err(format!(
                    "stream window {j} does not match its template window {tj}"
                ));
            }
            template[i] = t_time[tj];
            offset[i] = off;
        }
        let deps = dependencies(&specs, self.mode);
        let ready_layer: Vec<u32> = (0..n_s)
            .map(|i| {
                deps[i]
                    .iter()
                    .map(|&k| specs[k].b)
                    .fold(specs[i].b, u32::max)
            })
            .collect();
        let mut order: Vec<usize> = (0..n_s).collect();
        order.sort_by_key(|&i| (ready_layer[i], specs[i].phase.min(1), specs[i].commit.0));
        let mut suffix_min_a = vec![u32::MAX; n_s + 1];
        for k in (0..n_s).rev() {
            suffix_min_a[k] = suffix_min_a[k + 1].min(specs[order[k]].a);
        }
        Ok(StreamPlan {
            layers,
            specs,
            template,
            offset,
            order,
            ready_layer,
            deps,
            suffix_min_a,
        })
    }

    pub fn scratches(&self) -> Vec<Scratch> {
        self.template.scratches()
    }
}

/// 64 streams being decoded, one per bit of every detection word.
pub struct Stream<'a> {
    dec: &'a StreamDecoder,
    plan: &'a StreamPlan,
    correlated: bool,
    /// Live defect words per layer, from layer `base`.
    live: VecDeque<Vec<u64>>,
    base: u32,
    /// Defects in layers already let go: no window was left to explain them.
    dropped: usize,
    /// One lane's defects in the window being decoded, reused.
    defects: Vec<u32>,
    complete: u32,
    pos: usize,
    pub predictions: [u64; 64],
    /// Lane 0's decode time for each window, in plan order, if a clock was given.
    pub times: Vec<f64>,
    clock: Option<&'a dyn Fn() -> f64>,
    scratches: &'a mut [Scratch],
}

impl<'a> Stream<'a> {
    pub fn new(
        dec: &'a StreamDecoder,
        plan: &'a StreamPlan,
        correlated: bool,
        scratches: &'a mut [Scratch],
        clock: Option<&'a dyn Fn() -> f64>,
    ) -> Stream<'a> {
        let times = if clock.is_some() {
            vec![0.0; plan.specs.len()]
        } else {
            Vec::new()
        };
        Stream {
            dec,
            plan,
            correlated,
            live: VecDeque::new(),
            base: 0,
            dropped: 0,
            defects: Vec::new(),
            complete: 0,
            pos: 0,
            predictions: [0; 64],
            times,
            clock,
            scratches,
        }
    }

    fn layer_mut(&mut self, layer: u32, len: usize) -> &mut Vec<u64> {
        while self.base + self.live.len() as u32 <= layer {
            self.live.push_back(Vec::new());
        }
        let row = &mut self.live[(layer - self.base) as usize];
        if row.len() < len {
            row.resize(len, 0);
        }
        row
    }

    /// The next layer's detection words, in the order its detectors were made.
    pub fn push_layer(&mut self, words: &[u64]) -> Result<(), DecodeError> {
        let layer = self.complete;
        let row = self.layer_mut(layer, words.len());
        for (a, &w) in row.iter_mut().zip(words) {
            *a ^= w;
        }
        self.complete += 1;
        self.advance()
    }

    fn advance(&mut self) -> Result<(), DecodeError> {
        while self.pos < self.plan.order.len()
            && self.plan.ready_layer[self.plan.order[self.pos]] <= self.complete
        {
            let wi = self.plan.order[self.pos];
            self.decode_window(wi)?;
            self.pos += 1;
            // Let go of the layers no window still to come reads, counting any
            // defect left in them: nothing can explain it any more.
            let keep = self.plan.suffix_min_a[self.pos].min(self.complete);
            while self.base < keep {
                let Some(row) = self.live.pop_front() else {
                    break;
                };
                self.dropped += row.iter().map(|w| w.count_ones() as usize).sum::<usize>();
                self.base += 1;
            }
        }
        Ok(())
    }

    fn decode_window(&mut self, wi: usize) -> Result<(), DecodeError> {
        let (spec, ti, off) = (
            self.plan.specs[wi],
            self.plan.template[wi],
            self.plan.offset[wi],
        );
        let dec = self.dec;
        let w = &dec.template.windows[ti];
        let model = &dec.template.model;
        let boundary = model.num_detectors as u32;
        let (c0, c1) = (spec.commit.0 - off, spec.commit.1 - off);
        let mut defects = std::mem::take(&mut self.defects);
        for lane in 0..64 {
            let bit = 1u64 << lane;
            defects.clear();
            for (li, ranks) in dec.local[ti].iter().enumerate() {
                if let Some(row) = self.live.get((spec.a + li as u32 - self.base) as usize) {
                    for (r, &node) in ranks.iter().enumerate() {
                        if row.get(r).is_some_and(|&x| x & bit != 0) {
                            defects.push(node);
                        }
                    }
                }
            }
            defects.sort_unstable();
            let start = if lane == 0 {
                self.clock.map(|c| c())
            } else {
                None
            };
            let scratch = &mut self.scratches[ti];
            let edges = if self.correlated {
                w.window
                    .graph
                    .decode_correlated_edge_ids(&w.window.corr, scratch, &defects)?
            } else {
                w.window.graph.decode_edge_ids(scratch, &defects)?
            };
            if let (Some(t0), Some(c)) = (start, self.clock) {
                self.times[wi] = c() - t0;
            }
            for &le in &edges {
                let (u, v, _, o) = model.edges[w.window.global_edge[le as usize] as usize];
                let inside = |n: u32| n != boundary && (c0..c1).contains(&dec.place[n as usize].0);
                if inside(u) || inside(v) {
                    self.predictions[lane] ^= o;
                    for n in [u, v] {
                        if n != boundary {
                            let (tl, rank) = dec.place[n as usize];
                            let len = model.layers.members[tl as usize].len();
                            self.layer_mut(tl + off, len)[rank as usize] ^= bit;
                        }
                    }
                }
            }
        }
        self.defects = defects;
        Ok(())
    }

    /// Every window decoded; returns the defects left unexplained, over all 64
    /// streams: those in layers already let go, and those still held.
    pub fn finish(mut self) -> Result<(Self, usize), DecodeError> {
        self.advance()?;
        if self.pos != self.plan.order.len() {
            return Err(DecodeError::MatcherDeclined);
        }
        let held: usize = self
            .live
            .iter()
            .flatten()
            .map(|w| w.count_ones() as usize)
            .sum();
        let unexplained = self.dropped + held;
        Ok((self, unexplained))
    }
}

/// Sample 64 streams with `sampler` and decode them as they are made. Returns
/// the 64 predictions, the true observables, the defects left unexplained, and
/// lane 0's window decode times (with a clock).
pub fn run_stream(
    dec: &StreamDecoder,
    plan: &StreamPlan,
    sampler: &crate::batch_sampler::BatchSampler,
    rng: &mut crate::surface_code::Xorshift,
    correlated: bool,
    scratches: &mut [Scratch],
    clock: Option<&dyn Fn() -> f64>,
) -> Result<([u64; 64], u64, usize, Vec<f64>), DecodeError> {
    let mut stream = Stream::new(dec, plan, correlated, scratches, clock);
    let mut layer: Vec<u64> = Vec::new();
    let mut current = f64::NAN;
    let mut failure = None;
    let truth = sampler.run_timed(rng, &mut |_, t, w| {
        if t != current && !layer.is_empty() {
            if let Err(e) = stream.push_layer(&layer) {
                failure.get_or_insert(e);
            }
            layer.clear();
        }
        current = t;
        layer.push(w);
    });
    if let Some(e) = failure {
        return Err(e);
    }
    if !layer.is_empty() {
        stream.push_layer(&layer)?;
    }
    let (stream, unexplained) = stream.finish()?;
    Ok((
        stream.predictions,
        truth.first().copied().unwrap_or(0),
        unexplained,
        stream.times,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch_sampler::BatchSampler;
    use crate::dem::Dem;
    use crate::memory::{generate, generate_repeat, NoiseModel};
    use crate::surface_code::Xorshift;

    /// On a stream longer than its template, windows taken from the template
    /// decode every one of 64 streams exactly as windows cut from the full
    /// model of the stream.
    fn template_matches_full_model(d: usize, rounds: usize, commit: usize, buffer: usize) {
        let p = 0.008;
        let circuit = generate_repeat(CodeKind::Rotated, d, rounds, p, Basis::Z).unwrap();
        let sampler = BatchSampler::new(&circuit).unwrap();
        let full = Model::new(&Dem::from_circuit(&circuit).unwrap()).unwrap();
        let full_layers = full.layers.count();
        for mode in [Mode::Sliding, Mode::Parallel] {
            let dec = StreamDecoder::new(
                CodeKind::Rotated,
                d,
                p,
                Basis::Z,
                commit,
                buffer,
                mode,
                rounds,
            )
            .unwrap();
            assert!(
                dec.template_layers < dec.stream_layers,
                "{mode:?}: template of {} layers",
                dec.template_layers
            );
            let plan = dec.plan().unwrap();
            assert_eq!(plan.layers as usize, full_layers);
            let wd = WindowDecoder::new(
                Model::new(&Dem::from_circuit(&circuit).unwrap()).unwrap(),
                commit,
                buffer,
                mode,
            )
            .unwrap();
            let mut full_scratch = wd.scratches();
            for correlated in [false, true] {
                let mut rng = Xorshift::new(21);
                let batch = sampler.sample(&mut rng);
                let mut rng = Xorshift::new(21);
                let mut scratches = dec.scratches();
                let (pred, _, unexplained, _) = run_stream(
                    &dec,
                    &plan,
                    &sampler,
                    &mut rng,
                    correlated,
                    &mut scratches,
                    None,
                )
                .unwrap();
                assert_eq!(unexplained, 0, "{mode:?} correlated {correlated}");
                for (lane, &p) in pred.iter().enumerate() {
                    let want = wd
                        .decode_with(&batch.lane_defects(lane), correlated, &mut full_scratch)
                        .unwrap();
                    assert_eq!(want.unexplained, 0);
                    assert_eq!(p, want.observables, "d = {d} C = {commit} B = {buffer} {mode:?} correlated {correlated}, lane {lane}");
                }
            }
        }
    }

    #[test]
    fn a_template_decodes_exactly_as_the_full_model() {
        template_matches_full_model(3, 60, 3, 3);
    }

    /// The million-round configuration (d = 5, commit and buffer 5), and a
    /// commit shorter than the buffer, whose period differs.
    #[test]
    fn a_template_decodes_exactly_as_the_full_model_at_d5_and_unequal_windows() {
        template_matches_full_model(5, 100, 5, 5);
        template_matches_full_model(3, 60, 2, 3);
    }

    /// A defect in a layer no window will read again is counted when the
    /// layer is let go, not lost with it.
    #[test]
    fn defects_left_behind_are_counted() {
        let rounds = 30;
        let dec = StreamDecoder::new(
            CodeKind::Rotated,
            3,
            0.001,
            Basis::Z,
            3,
            3,
            Mode::Sliding,
            rounds,
        )
        .unwrap();
        let plan = dec.plan().unwrap();
        let members = &dec.template.model.layers.members;
        let mut scratches = dec.scratches();
        let mut stream = Stream::new(&dec, &plan, false, &mut scratches, None);
        // Skip every window but the last, so layer 0's defect is never read.
        stream.pos = plan.order.len() - 1;
        for l in 0..plan.layers as usize {
            let len = members[(l).min(members.len() - 1)].len();
            let mut row = vec![0u64; len];
            if l == 0 {
                row[0] = 1 << 5;
            }
            stream.push_layer(&row).unwrap();
        }
        assert!(
            stream.live.is_empty(),
            "every layer let go after the last window"
        );
        let (_, unexplained) = stream.finish().unwrap();
        assert_eq!(unexplained, 1);
    }

    /// The bulk windows repeat: at three offsets deep in a long model, the
    /// windows are the same graph once shifted.
    #[test]
    fn bulk_windows_are_the_same_graph() {
        let circuit = generate(
            CodeKind::Rotated,
            3,
            60,
            NoiseModel::Sd6 { p: 0.004 },
            Basis::Z,
        )
        .unwrap();
        let model = Model::new(&Dem::from_circuit(&circuit).unwrap()).unwrap();
        let signature = |a: u32| {
            let w = model.window(a, a + 6, true, true);
            let g = &w.graph;
            (
                g.num_nodes,
                g.to.clone(),
                g.w.clone(),
                g.obs.clone(),
                (0..g.num_edges() as u32)
                    .map(|e| w.corr.rules_of(e))
                    .collect::<Vec<_>>(),
            )
        };
        let first = signature(15);
        assert_eq!(signature(27), first);
        assert_eq!(signature(40), first);
        assert_ne!(
            signature(0),
            first,
            "the first rounds differ, which is why the ends come from the template's ends"
        );
    }
}

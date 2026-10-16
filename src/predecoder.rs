//! The AI pre-decoder: a fully convolutional 3D network over the detectors' space-time grid, in
//! front of a global decoder, in the shape of NVIDIA's (Chamberland, Olle, Li, Thornton and
//! Baratta, arXiv 2604.12841).
//!
//! WHY THIS EXISTS
//! ---------------
//! Most faults at a useful error rate are isolated: one fault, its one or two detection events,
//! nothing near. A small convolution recognises them, and the correction it applies removes them
//! from the syndrome before matching runs, so the global decoder sees a sparser syndrome (faster)
//! and, where the network has learnt what matching cannot see (the X and Z halves of a Y fault
//! firing together, a hook's two edges), a better one. The network's receptive field is local, so
//! a network trained on a small volume runs on any distance and any number of rounds.
//!
//! The paper's network reads the X and Z detection events on a grid of the rotated surface code's
//! data qubits and predicts data-qubit corrections and timelike syndrome flips. Here the same
//! network reads any decomposed model whose detectors have coordinates (x, y, t): the detectors
//! sit on the grid their coordinates make, one input channel per sector (the components the
//! model's graph-like faults connect: a CSS memory's X and Z detectors) for its events and one
//! for where its detectors are, and the outputs are the model's graph-like faults themselves, one
//! channel per kind (its sector and the offset from its first detector to its second, or the
//! boundary), at its first detector. A spacelike correction is a kind joining two detectors of a
//! round, a timelike one a kind joining a detector to itself a round later. Each fault the network
//! is confident of (above the threshold) is applied: its detection events flipped, its
//! observables flipped in the shot's prediction. The global decoder decodes what is left, and the
//! prediction is its correction's observables and the pre-decoder's together.
//!
//! The network is the paper's: 3D convolutions (stride 1, zero padding to keep the grid), GELU
//! (tanh) between, a sigmoid out, trained by binary cross-entropy per fault kind per voxel
//! against the faults that fired (from the model's own sampler), with Lion or Adam, warmup and
//! the paper's decay. Labels are the faults' graph-like parts: two faults with the same symptom
//! are one kind, and of the faults that cannot be told apart the network learns the probability,
//! so where it cannot know it stays below the threshold and leaves the events to the global
//! decoder.

use std::collections::{BTreeMap, HashMap};

use crate::dem::Dem;
use crate::dem_sampler::DemSampler;
use crate::surface_code::Xorshift;

/// No fault of a kind at a voxel.
const NONE: u32 = u32::MAX;

/// A kind of graph-like fault: its sector, and the offset from its first detector (in time, then
/// space) to its second, or the boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Kind {
    pub sector: u32,
    pub boundary: bool,
    pub dt: i32,
    pub dy: i32,
    pub dx: i32,
}

/// A graph-like fault the network can apply: its detectors (one for a boundary fault), its
/// observables, and its probability (the faults with this symptom together).
#[derive(Clone, Debug)]
pub struct Fault {
    pub detectors: Vec<u32>,
    pub observables: u64,
    pub p: f64,
}

/// Where each detector sits on the grid, and which faults the outputs are.
#[derive(Clone, Debug)]
pub struct Layout {
    /// (t, y, x).
    pub dims: [usize; 3],
    pub num_sectors: usize,
    pub num_detectors: usize,
    pub num_observables: usize,
    /// Per detector, its voxel and sector.
    pub voxel: Vec<u32>,
    pub sector: Vec<u32>,
    pub kinds: Vec<Kind>,
    pub faults: Vec<Fault>,
    /// Per voxel and kind, the fault there (or `NONE`).
    pub slot: Vec<u32>,
    /// Per mechanism of the model, the slots of its graph-like parts.
    pub parts: Vec<Vec<u32>>,
    /// The input with no events: the sectors' presence channels.
    pub blank: Vec<f32>,
}

/// Each coordinate's index on the regular grid the values make.
fn grid(values: &[f64], axis: &str) -> Result<(Vec<usize>, usize), String> {
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    let lo = sorted[0];
    let step = sorted.windows(2).map(|w| w[1] - w[0]).fold(f64::INFINITY, f64::min);
    if sorted.len() == 1 {
        return Ok((vec![0; values.len()], 1));
    }
    let mut out = Vec::with_capacity(values.len());
    for &v in values {
        let i = (v - lo) / step;
        if (i - i.round()).abs() > 1e-6 {
            return Err(format!("the detectors' {axis} coordinates are not on a regular grid (step {step}, and {v})"));
        }
        out.push(i.round() as usize);
    }
    let n = out.iter().max().unwrap() + 1;
    Ok((out, n))
}

fn find(parent: &mut [u32], mut a: u32) -> u32 {
    while parent[a as usize] != a {
        parent[a as usize] = parent[parent[a as usize] as usize];
        a = parent[a as usize];
    }
    a
}

/// A mechanism's graph-like parts: its pieces, or itself when it has none and is graph-like.
fn graphlike(m: &crate::dem::Mechanism) -> Vec<(Vec<u32>, u64)> {
    if !m.pieces.is_empty() {
        m.pieces.iter().filter(|p| !p.detectors.is_empty()).map(|p| (p.detectors.clone(), p.observables)).collect()
    } else if (1..=2).contains(&m.detectors.len()) {
        vec![(m.detectors.clone(), m.observables)]
    } else {
        Vec::new()
    }
}

impl Layout {
    /// The layout of a decomposed model whose detectors all have coordinates (x, y, t).
    /// `sectors`: per detector, its sector (by default, the components its graph-like faults
    /// connect, numbered in order of their first detector).
    pub fn new(dem: &Dem, sectors: Option<&[u32]>) -> Result<Layout, String> {
        let nd = dem.num_detectors;
        if nd == 0 {
            return Err("the model has no detectors".into());
        }
        let mut axes: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for d in 0..nd {
            let c = dem.detector_coords.get(d).map(|c| c.as_slice()).unwrap_or(&[]);
            if c.len() < 3 {
                return Err(format!("the pre-decoder needs every detector's coordinates (x, y, t); D{d} has {}", c.len()));
            }
            axes[0].push(c[2]);
            axes[1].push(c[1]);
            axes[2].push(c[0]);
        }
        let (t, nt) = grid(&axes[0], "t")?;
        let (y, ny) = grid(&axes[1], "y")?;
        let (x, nx) = grid(&axes[2], "x")?;
        let dims = [nt, ny, nx];
        let voxel: Vec<u32> = (0..nd).map(|d| ((t[d] * ny + y[d]) * nx + x[d]) as u32).collect();
        let parts_of: Vec<Vec<(Vec<u32>, u64)>> = dem.mechanisms.iter().map(graphlike).collect();
        let sector: Vec<u32> = match sectors {
            Some(s) => {
                if s.len() != nd {
                    return Err(format!("{} sectors for {nd} detectors", s.len()));
                }
                s.to_vec()
            }
            None => {
                let mut parent: Vec<u32> = (0..nd as u32).collect();
                for parts in &parts_of {
                    for (dets, _) in parts {
                        if let [a, b] = dets.as_slice() {
                            let (ra, rb) = (find(&mut parent, *a), find(&mut parent, *b));
                            if ra != rb {
                                parent[ra.max(rb) as usize] = ra.min(rb);
                            }
                        }
                    }
                }
                // Components numbered by their first detector; a detector no fault joins to
                // another is in sector 0.
                let mut joined = vec![false; nd];
                for parts in &parts_of {
                    for (dets, _) in parts {
                        if dets.len() == 2 {
                            joined[dets[0] as usize] = true;
                            joined[dets[1] as usize] = true;
                        }
                    }
                }
                let mut number: HashMap<u32, u32> = HashMap::new();
                (0..nd as u32)
                    .map(|d| {
                        if !joined[d as usize] {
                            return 0;
                        }
                        let r = find(&mut parent, d);
                        let next = number.len() as u32;
                        *number.entry(r).or_insert(next)
                    })
                    .collect()
            }
        };
        let num_sectors = sector.iter().max().map_or(1, |&s| s as usize + 1);
        let mut seen: HashMap<(u32, u32), u32> = HashMap::new();
        for d in 0..nd as u32 {
            if let Some(o) = seen.insert((voxel[d as usize], sector[d as usize]), d) {
                return Err(format!("D{o} and D{d} are in one sector at the same coordinates"));
            }
        }
        // Each graph-like symptom (detectors, observables) with its probability.
        let mut symptom: BTreeMap<(Vec<u32>, u64), f64> = BTreeMap::new();
        for (m, parts) in dem.mechanisms.iter().zip(&parts_of) {
            for (dets, obs) in parts {
                let mut dets = dets.clone();
                dets.sort_unstable();
                let p = symptom.entry((dets, *obs)).or_insert(0.0);
                *p = *p * (1.0 - m.p) + m.p * (1.0 - *p);
            }
        }
        // Its kind and voxel; at a kind and voxel, the most likely symptom.
        let place = |dets: &[u32]| -> (u32, Kind) {
            let mut ds: Vec<u32> = dets.to_vec();
            ds.sort_by_key(|&d| voxel[d as usize]);
            let a = ds[0] as usize;
            let (ta, ya, xa) = (t[a] as i32, y[a] as i32, x[a] as i32);
            let kind = match ds.get(1) {
                None => Kind { sector: sector[a], boundary: true, dt: 0, dy: 0, dx: 0 },
                Some(&b) => {
                    let b = b as usize;
                    Kind { sector: sector[a], boundary: false, dt: t[b] as i32 - ta, dy: y[b] as i32 - ya, dx: x[b] as i32 - xa }
                }
            };
            (voxel[a], kind)
        };
        let mut best: BTreeMap<(Kind, u32), (Vec<u32>, u64, f64)> = BTreeMap::new();
        for ((dets, obs), p) in &symptom {
            let (v, kind) = place(dets);
            let e = best.entry((kind, v)).or_insert((dets.clone(), *obs, -1.0));
            if *p > e.2 {
                *e = (dets.clone(), *obs, *p);
            }
        }
        let mut kinds: Vec<Kind> = best.keys().map(|(k, _)| *k).collect();
        kinds.dedup();
        let kind_index: HashMap<Kind, u32> = kinds.iter().enumerate().map(|(i, k)| (*k, i as u32)).collect();
        let nk = kinds.len();
        let nv = nt * ny * nx;
        // Faults in the order of their slots: by voxel, then kind.
        let mut placed: Vec<(usize, Fault)> = best.into_iter().map(|((kind, v), (dets, obs, p))| (v as usize * nk + kind_index[&kind] as usize, Fault { detectors: dets, observables: obs, p })).collect();
        placed.sort_by_key(|x| x.0);
        let mut slot = vec![NONE; nv * nk];
        let mut faults = Vec::with_capacity(placed.len());
        for (i, f) in placed {
            slot[i] = faults.len() as u32;
            faults.push(f);
        }
        let parts = parts_of
            .iter()
            .map(|ps| {
                ps.iter()
                    .map(|(dets, _)| {
                        let (v, kind) = place(dets);
                        v * nk as u32 + kind_index[&kind]
                    })
                    .collect()
            })
            .collect();
        let cin = 2 * num_sectors;
        let mut blank = vec![0.0f32; nv * cin];
        for d in 0..nd {
            blank[voxel[d] as usize * cin + num_sectors + sector[d] as usize] = 1.0;
        }
        Ok(Layout { dims, num_sectors, num_detectors: nd, num_observables: dem.num_observables, voxel, sector, kinds, faults, slot, parts, blank })
    }

    pub fn num_voxels(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn num_inputs(&self) -> usize {
        2 * self.num_sectors
    }

    pub fn num_kinds(&self) -> usize {
        self.kinds.len()
    }

    /// A shot's input (one packed row of detection events).
    pub fn encode(&self, row: &[u8], x: &mut [f32]) {
        x.copy_from_slice(&self.blank);
        let cin = self.num_inputs();
        for (d, (&v, &s)) in self.voxel.iter().zip(&self.sector).enumerate() {
            if row[d / 8] >> (d % 8) & 1 == 1 {
                x[v as usize * cin + s as usize] = 1.0;
            }
        }
    }

    /// A shot's labels (one packed row of the model's faults that fired): per voxel and kind,
    /// whether its fault's parts fired an odd number of times.
    pub fn labels(&self, row: &[u8], y: &mut [f32]) {
        y.fill(0.0);
        for (m, parts) in self.parts.iter().enumerate() {
            if row[m / 8] >> (m % 8) & 1 == 1 {
                for &s in parts {
                    y[s as usize] = 1.0 - y[s as usize];
                }
            }
        }
    }
}

/* -- The network ----------------------------------------------------------- */

/// 3D convolutions with the same kernel, GELU between; parameters flat, each layer's weights
/// (kernel offset, input channel, output channel) then its biases.
#[derive(Clone, Debug)]
pub struct Net {
    pub channels: Vec<usize>,
    pub kernel: usize,
    pub params: Vec<f32>,
}

const SQRT_2_OVER_PI: f32 = 0.797_884_6;

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
}

fn gelu_grad(x: f32) -> f32 {
    let th = (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh();
    0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * x * x)
}

/// Per shot, the activations a forward pass keeps for the backward one, and the grid's
/// neighbours: per voxel and kernel offset, the voxel there (or `NONE` off the grid).
pub struct Work {
    nb: Vec<u32>,
    /// Per layer, its input (after the previous layer's GELU) and its pre-activation output.
    acts: Vec<Vec<f32>>,
    pre: Vec<Vec<f32>>,
    grad_out: Vec<f32>,
    grad_in: Vec<f32>,
    transposed: Vec<f32>,
}

impl Net {
    /// `channels`: inputs, each hidden layer's filters, outputs. Weights uniform in ±1/√fan-in
    /// (PyTorch's default), biases likewise but the last layer's, which start at −4 (outputs
    /// near 2%, as faults are rare).
    pub fn new(channels: &[usize], kernel: usize, seed: u64) -> Result<Net, String> {
        if channels.len() < 2 || channels.contains(&0) {
            return Err("a network needs at least an input and an output layer, none empty".into());
        }
        if kernel % 2 == 0 {
            return Err(format!("the kernel must be odd, not {kernel}"));
        }
        let k3 = kernel * kernel * kernel;
        let mut rng = Xorshift::new(seed ^ 0x5DEE_CE66_D1CE_4E5B);
        let mut params = Vec::new();
        for (l, w) in channels.windows(2).enumerate() {
            let bound = 1.0 / ((w[0] * k3) as f64).sqrt();
            for _ in 0..k3 * w[0] * w[1] {
                params.push(((rng.next_f64() * 2.0 - 1.0) * bound) as f32);
            }
            let last = l + 2 == channels.len();
            for _ in 0..w[1] {
                let b = (rng.next_f64() * 2.0 - 1.0) * bound;
                params.push(if last { -4.0 } else { b as f32 });
            }
        }
        Ok(Net { channels: channels.to_vec(), kernel, params })
    }

    pub fn num_layers(&self) -> usize {
        self.channels.len() - 1
    }

    /// Layer `l`'s (weight offset, bias offset).
    fn offsets(&self, l: usize) -> (usize, usize) {
        let k3 = self.kernel.pow(3);
        let mut o = 0;
        for w in self.channels.windows(2).take(l) {
            o += k3 * w[0] * w[1] + w[1];
        }
        (o, o + k3 * self.channels[l] * self.channels[l + 1])
    }

    pub fn work(&self, dims: [usize; 3]) -> Work {
        let voxels: usize = dims.iter().product();
        let n = self.num_layers();
        Work {
            nb: neighbours(dims, self.kernel),
            acts: (0..n).map(|l| vec![0.0; voxels * self.channels[l]]).collect(),
            pre: (0..n).map(|l| vec![0.0; voxels * self.channels[l + 1]]).collect(),
            grad_out: Vec::new(),
            grad_in: Vec::new(),
            transposed: Vec::new(),
        }
    }

    /// The outputs (logits, voxel by kind) for the input `x`, left in the work's last layer.
    pub fn forward<'w>(&self, x: &[f32], w: &'w mut Work) -> &'w [f32] {
        let k3 = self.kernel.pow(3);
        w.acts[0].copy_from_slice(x);
        for l in 0..self.num_layers() {
            let (wo, bo) = self.offsets(l);
            let (cin, cout) = (self.channels[l], self.channels[l + 1]);
            conv(&w.nb, k3, cin, cout, &self.params[wo..bo], Some(&self.params[bo..bo + cout]), false, &w.acts[l], &mut w.pre[l]);
            if l + 1 < self.num_layers() {
                for (a, &p) in w.acts[l + 1].iter_mut().zip(&w.pre[l]) {
                    *a = gelu(p);
                }
            }
        }
        &w.pre[self.num_layers() - 1]
    }

    /// Adds to `grad` the gradient of the loss whose gradient by the outputs is `dout`, after a
    /// forward pass in `w`.
    pub fn backward(&self, dout: &[f32], w: &mut Work, grad: &mut [f32]) {
        let k3 = self.kernel.pow(3);
        w.grad_out.clear();
        w.grad_out.extend_from_slice(dout);
        for l in (0..self.num_layers()).rev() {
            let (wo, bo) = self.offsets(l);
            let (cin, cout) = (self.channels[l], self.channels[l + 1]);
            let (gw, gb) = grad[wo..bo + cout].split_at_mut(bo - wo);
            conv_weight_grad(&w.nb, k3, cin, cout, &w.acts[l], &w.grad_out, gw, gb);
            if l == 0 {
                break;
            }
            // The input's gradient is the convolution of the output's by the mirrored kernel,
            // its weights transposed (output channel, input channel).
            let wt = &self.params[wo..bo];
            w.transposed.resize(wt.len(), 0.0);
            for off in 0..k3 {
                for ci in 0..cin {
                    for co in 0..cout {
                        w.transposed[(off * cout + co) * cin + ci] = wt[(off * cin + ci) * cout + co];
                    }
                }
            }
            w.grad_in.resize(w.acts[l].len(), 0.0);
            conv(&w.nb, k3, cout, cin, &w.transposed, None, true, &w.grad_out, &mut w.grad_in);
            for (g, &p) in w.grad_in.iter_mut().zip(&w.pre[l - 1]) {
                *g *= gelu_grad(p);
            }
            std::mem::swap(&mut w.grad_out, &mut w.grad_in);
        }
    }
}

/// Per voxel and kernel offset (t, y, x, each 0..k), the voxel there, or `NONE` off the grid.
/// Symmetric: u is v's neighbour at an offset when v is u's at the mirrored one.
fn neighbours(dims: [usize; 3], k: usize) -> Vec<u32> {
    let h = (k / 2) as isize;
    let [nt, ny, nx] = dims.map(|d| d as isize);
    let mut nb = Vec::with_capacity((nt * ny * nx) as usize * k.pow(3));
    for t in 0..nt {
        for y in 0..ny {
            for x in 0..nx {
                for kt in 0..k as isize {
                    for ky in 0..k as isize {
                        for kx in 0..k as isize {
                            let (tt, yy, xx) = (t + kt - h, y + ky - h, x + kx - h);
                            let inside = (0..nt).contains(&tt) && (0..ny).contains(&yy) && (0..nx).contains(&xx);
                            nb.push(if inside { ((tt * ny + yy) * nx + xx) as u32 } else { NONE });
                        }
                    }
                }
            }
        }
    }
    nb
}

/// Output channels a register block holds.
const B: usize = 16;

/// out[v, :] = bias + Σ over offsets and input channels of input[neighbour, ci] · w[offset, ci, :]
/// (the offset mirrored with `mirror`), a block of output channels at a time in registers.
#[allow(clippy::too_many_arguments)]
fn conv(nb: &[u32], k3: usize, cin: usize, cout: usize, w: &[f32], bias: Option<&[f32]>, mirror: bool, input: &[f32], out: &mut [f32]) {
    for (nbv, o) in nb.chunks_exact(k3).zip(out.chunks_exact_mut(cout)) {
        let mut c0 = 0;
        while c0 < cout {
            let r = (cout - c0).min(B);
            let mut acc = [0.0f32; B];
            if let Some(b) = bias {
                acc[..r].copy_from_slice(&b[c0..c0 + r]);
            }
            for (off, &u) in nbv.iter().enumerate() {
                if u == NONE {
                    continue;
                }
                let base = (if mirror { k3 - 1 - off } else { off }) * cin * cout + c0;
                let inp = &input[u as usize * cin..(u as usize + 1) * cin];
                if r == B {
                    for (ci, &a) in inp.iter().enumerate() {
                        if a != 0.0 {
                            let row: &[f32; B] = w[base + ci * cout..base + ci * cout + B].try_into().unwrap();
                            for j in 0..B {
                                acc[j] += a * row[j];
                            }
                        }
                    }
                } else {
                    for (ci, &a) in inp.iter().enumerate() {
                        if a != 0.0 {
                            for (x, &ww) in acc[..r].iter_mut().zip(&w[base + ci * cout..base + ci * cout + r]) {
                                *x += a * ww;
                            }
                        }
                    }
                }
            }
            o[c0..c0 + r].copy_from_slice(&acc[..r]);
            c0 += r;
        }
    }
}

/// gw[offset, ci, :] += Σ_v input[neighbour of v at offset, ci] · g[v, :], and gb += Σ_v g[v, :].
#[allow(clippy::too_many_arguments)]
fn conv_weight_grad(nb: &[u32], k3: usize, cin: usize, cout: usize, input: &[f32], g: &[f32], gw: &mut [f32], gb: &mut [f32]) {
    for gv in g.chunks_exact(cout) {
        for (b, &x) in gb.iter_mut().zip(gv) {
            *b += x;
        }
    }
    let nv = nb.len() / k3;
    for off in 0..k3 {
        for ci in 0..cin {
            let mut c0 = 0;
            while c0 < cout {
                let r = (cout - c0).min(B);
                let mut acc = [0.0f32; B];
                for v in 0..nv {
                    let u = nb[v * k3 + off];
                    if u == NONE {
                        continue;
                    }
                    let a = input[u as usize * cin + ci];
                    if a == 0.0 {
                        continue;
                    }
                    let row = &g[v * cout + c0..v * cout + c0 + r];
                    if r == B {
                        let row: &[f32; B] = row.try_into().unwrap();
                        for j in 0..B {
                            acc[j] += a * row[j];
                        }
                    } else {
                        for (x, &gg) in acc[..r].iter_mut().zip(row) {
                            *x += a * gg;
                        }
                    }
                }
                for (x, &a) in gw[(off * cin + ci) * cout + c0..(off * cin + ci) * cout + c0 + r].iter_mut().zip(&acc[..r]) {
                    *x += a;
                }
                c0 += r;
            }
        }
    }
}

/* -- Loss, training ------------------------------------------------------- */

/// Binary cross-entropy from logits, summed over the voxels and kinds that have a fault, and its
/// gradient by the logits (zero elsewhere).
pub fn bce(logits: &[f32], labels: &[f32], slot: &[u32], grad: &mut [f32]) -> f64 {
    let mut loss = 0.0f64;
    for (((&z, &y), &s), g) in logits.iter().zip(labels).zip(slot).zip(grad.iter_mut()) {
        if s == NONE {
            *g = 0.0;
            continue;
        }
        loss += (z.max(0.0) - z * y + (-z.abs()).exp().ln_1p()) as f64;
        *g = 1.0 / (1.0 + (-z).exp()) - y;
    }
    loss
}

/// Shots whose gradients one thread sums before the blocks are summed in order, so that the
/// result does not depend on the threads.
const BLOCK: usize = 8;

/// The mean loss of `shots` shots (detection events and the faults that fired, packed rows) and
/// its gradient by the parameters; the same on any number of threads.
pub fn loss_and_grad(layout: &Layout, net: &Net, dets: &[u8], errors: &[u8], shots: usize, threads: usize) -> (f64, Vec<f32>) {
    let (sd, se) = (layout.num_detectors.div_ceil(8), layout.parts.len().div_ceil(8));
    let nv = layout.num_voxels();
    let blocks = crate::parallel::parallel(shots.div_ceil(BLOCK), threads, |range| {
        let mut work = net.work(layout.dims);
        let mut x = vec![0.0f32; nv * layout.num_inputs()];
        let mut y = vec![0.0f32; nv * layout.num_kinds()];
        let mut g = vec![0.0f32; y.len()];
        range
            .map(|b| {
                let mut grad = vec![0.0f32; net.params.len()];
                let mut loss = 0.0;
                for s in b * BLOCK..((b + 1) * BLOCK).min(shots) {
                    layout.encode(&dets[s * sd..(s + 1) * sd], &mut x);
                    layout.labels(&errors[s * se..(s + 1) * se], &mut y);
                    let z = net.forward(&x, &mut work);
                    loss += bce(z, &y, &layout.slot, &mut g);
                    net.backward(&g, &mut work, &mut grad);
                }
                (loss, grad)
            })
            .collect()
    });
    let mut total = vec![0.0f32; net.params.len()];
    let mut loss = 0.0;
    for (l, g) in blocks {
        loss += l;
        for (t, x) in total.iter_mut().zip(g) {
            *t += x;
        }
    }
    let n = shots.max(1) as f32;
    for t in &mut total {
        *t /= n;
    }
    (loss / shots.max(1) as f64, total)
}

/// Lion (the paper's: β1 0.9, β2 0.95, decoupled weight decay) or Adam (β 0.9, 0.999).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Optimizer {
    Lion { weight_decay: f32 },
    Adam,
}

pub struct Trainer {
    pub optimizer: Optimizer,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub step: u64,
}

impl Trainer {
    pub fn new(optimizer: Optimizer, num_params: usize) -> Trainer {
        Trainer { optimizer, m: vec![0.0; num_params], v: if optimizer == Optimizer::Adam { vec![0.0; num_params] } else { Vec::new() }, step: 0 }
    }

    /// One step of size `lr`.
    pub fn update(&mut self, params: &mut [f32], grad: &[f32], lr: f32) {
        self.step += 1;
        match self.optimizer {
            Optimizer::Lion { weight_decay } => {
                for ((p, m), &g) in params.iter_mut().zip(&mut self.m).zip(grad) {
                    let c = 0.9 * *m + 0.1 * g;
                    let u = if c > 0.0 {
                        1.0
                    } else if c < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    *p -= lr * (u + weight_decay * *p);
                    *m = 0.95 * *m + 0.05 * g;
                }
            }
            Optimizer::Adam => {
                let t = self.step as i32;
                let (c1, c2) = (1.0 - 0.9f32.powi(t), 1.0 - 0.999f32.powi(t));
                for (((p, m), v), &g) in params.iter_mut().zip(&mut self.m).zip(&mut self.v).zip(grad) {
                    *m = 0.9 * *m + 0.1 * g;
                    *v = 0.999 * *v + 0.001 * g * g;
                    *p -= lr * (*m / c1) / ((*v / c2).sqrt() + 1e-8);
                }
            }
        }
    }
}

/// The paper's schedule: linear warmup over `warmup` steps, then the rate ×0.7 at a quarter and
/// at half of the `steps`.
pub fn learning_rate(base: f32, step: usize, steps: usize, warmup: usize) -> f32 {
    let mut lr = base;
    if step < warmup {
        lr *= (step + 1) as f32 / warmup as f32;
    }
    if 4 * step >= steps {
        lr *= 0.7;
    }
    if 2 * step >= steps {
        lr *= 0.7;
    }
    lr
}

/// Trains on fresh shots of the model each step (`batch` of them, the sampler's batches
/// `seed`-streamed by step), returning each step's mean loss.
#[allow(clippy::too_many_arguments)]
pub fn train(layout: &Layout, net: &mut Net, trainer: &mut Trainer, sampler: &DemSampler, steps: usize, batch: usize, lr: f32, warmup: usize, seed: u64, threads: usize) -> Vec<f64> {
    let per = batch.div_ceil(64) as u64;
    let mut losses = Vec::with_capacity(steps);
    for step in 0..steps {
        let first = (trainer.step + 1) * per;
        let (dets, _, errors) = sampler.sample_seeded(seed, first, batch, threads, true);
        let (loss, grad) = loss_and_grad(layout, net, &dets, &errors, batch, threads);
        trainer.update(&mut net.params, &grad, learning_rate(lr, step, steps, warmup));
        losses.push(loss);
    }
    losses
}

/* -- Inference ------------------------------------------------------------ */

/// Per shot: the detection events left (packed row), the observables of the faults applied, and
/// how many were applied.
pub fn predecode(layout: &Layout, net: &Net, dets: &[u8], shots: usize, threshold: f32, threads: usize) -> (Vec<u8>, Vec<u64>, Vec<u32>) {
    let sd = layout.num_detectors.div_ceil(8);
    let nv = layout.num_voxels();
    let logit = (threshold / (1.0 - threshold)).ln();
    let parts = crate::parallel::parallel(shots, threads, |range| {
        let mut work = net.work(layout.dims);
        let mut x = vec![0.0f32; nv * layout.num_inputs()];
        let mut out = Vec::with_capacity(range.len());
        for s in range {
            let row = &dets[s * sd..(s + 1) * sd];
            let mut left = row.to_vec();
            let (mut obs, mut applied) = (0u64, 0u32);
            if row.iter().any(|&b| b != 0) {
                layout.encode(row, &mut x);
                let z = net.forward(&x, &mut work);
                for (i, &zi) in z.iter().enumerate() {
                    let f = layout.slot[i];
                    if zi > logit && f != NONE {
                        let fault = &layout.faults[f as usize];
                        for &d in &fault.detectors {
                            left[d as usize / 8] ^= 1 << (d % 8);
                        }
                        obs ^= fault.observables;
                        applied += 1;
                    }
                }
            }
            out.push((left, obs, applied));
        }
        out
    });
    let mut left = Vec::with_capacity(shots * sd);
    let (mut obs, mut applied) = (Vec::with_capacity(shots), Vec::with_capacity(shots));
    for (l, o, a) in parts {
        left.extend(l);
        obs.push(o);
        applied.push(a);
    }
    (left, obs, applied)
}

/// The network's probabilities as a global decoder's priors: per shot, each fault the network
/// sees gets its probability (belief-matching's move, with the network in BP's place), and the
/// decoder decodes the whole syndrome.
pub struct Soft {
    pub dec: crate::reweight::PriorDecoder,
    /// Per fault of the layout, the decoder's (or `NONE`).
    pub fault: Vec<u32>,
}

impl Soft {
    /// For a graph decoder (matching, union-find): the layout's faults are its edges.
    pub fn new(layout: &Layout, dec: crate::reweight::PriorDecoder) -> Soft {
        let nd = layout.num_detectors as u32;
        let fault = layout
            .faults
            .iter()
            .map(|f| match f.detectors.as_slice() {
                [a] => dec.edge(*a, nd),
                [a, b] => dec.edge(*a, *b),
                _ => None,
            }.unwrap_or(NONE))
            .collect();
        Soft { dec, fault }
    }
}

/// Per shot, the observables of the faults the network applied (those above `threshold`) and of
/// the decoder's correction of what is left, on the network's priors: q for a fault it did not
/// apply, 1 − q for one it did (the chance that applying it was wrong). At threshold 1/2 this is
/// the decoder on the whole syndrome with weights ln((1 − q)/q), negative ones included (the
/// faults applied), as matching takes negative weights; and the decoder sees a sparser syndrome.
/// `None` where no correction explains the shot.
pub fn decode_soft(layout: &Layout, net: &Net, soft: &Soft, dets: &[u8], shots: usize, threshold: f32, threads: usize) -> Vec<Option<u64>> {
    let nd = layout.num_detectors;
    let sd = nd.div_ceil(8);
    let nv = layout.num_voxels();
    let logit = (threshold / (1.0 - threshold)).ln();
    crate::parallel::parallel(shots, threads, |range| {
        let mut work = net.work(layout.dims);
        let mut dw = soft.dec.work();
        let mut x = vec![0.0f32; nv * layout.num_inputs()];
        let mut defects = Vec::new();
        let mut syndrome = vec![0u8; nd];
        range
            .map(|s| {
                let row = &dets[s * sd..(s + 1) * sd];
                if row.iter().all(|&b| b == 0) {
                    return Some(0);
                }
                syndrome.fill(0);
                for (d, v) in syndrome.iter_mut().enumerate() {
                    *v = row[d / 8] >> (d % 8) & 1;
                }
                layout.encode(row, &mut x);
                let z = net.forward(&x, &mut work);
                let mut obs = 0u64;
                for (i, &zi) in z.iter().enumerate() {
                    let f = layout.slot[i];
                    if f == NONE {
                        continue;
                    }
                    let q = 1.0 / (1.0 + (-zi as f64).exp());
                    let applied = zi > logit;
                    if applied {
                        let fault = &layout.faults[f as usize];
                        for &d in &fault.detectors {
                            syndrome[d as usize] ^= 1;
                        }
                        obs ^= fault.observables;
                    }
                    if soft.fault[f as usize] != NONE {
                        soft.dec.set(&mut dw, soft.fault[f as usize], if applied { 1.0 - q } else { q });
                    }
                }
                defects.clear();
                defects.extend(syndrome.iter().enumerate().filter(|x| *x.1 != 0).map(|x| x.0 as u32));
                let out = if defects.is_empty() { Some(0) } else { soft.dec.round(&mut dw, &defects, &syndrome, 0).map(|o| o.0) };
                soft.dec.restore(&mut dw);
                out.map(|o| o ^ obs)
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dem() -> Dem {
        // A repetition code over two rounds: detectors at x = 0, 2 and t = 0, 1.
        Dem::parse(
            "error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1\nerror(0.05) D0 D2\nerror(0.05) D1 D3\nerror(0.1) D2 L0\nerror(0.1) D2 D3\nerror(0.1) D3\n\
             detector(0, 0, 0) D0\ndetector(2, 0, 0) D1\ndetector(0, 0, 1) D2\ndetector(2, 0, 1) D3",
        )
        .unwrap()
    }

    #[test]
    fn layout_places_faults_by_kind() {
        let l = Layout::new(&dem(), None).unwrap();
        assert_eq!(l.dims, [2, 1, 2]);
        assert_eq!(l.num_sectors, 1);
        // Kinds: the boundary, a step in x, a step in t.
        assert_eq!(l.num_kinds(), 3);
        assert_eq!(l.faults.len(), 8);
        // D0 L0 and D1 are both boundary faults, at different voxels.
        let b = l.kinds.iter().position(|k| k.boundary).unwrap();
        let f = l.slot[b];
        assert_eq!(l.faults[f as usize].detectors, vec![0]);
        assert_eq!(l.faults[f as usize].observables, 1);
    }

    #[test]
    fn the_gradient_is_the_losss() {
        let d = dem();
        let l = Layout::new(&d, None).unwrap();
        let mut net = Net::new(&[l.num_inputs(), 3, l.num_kinds()], 3, 7).unwrap();
        for p in net.params.iter_mut() {
            *p *= 3.0;
        }
        let s = DemSampler::new(&d);
        let (dets, _, errors) = s.sample_seeded(1, 0, 16, 1, true);
        let (_, grad) = loss_and_grad(&l, &net, &dets, &errors, 16, 1);
        for i in (0..net.params.len()).step_by(5) {
            let mut a = net.clone();
            a.params[i] += 1e-2;
            let mut b = net.clone();
            b.params[i] -= 1e-2;
            let numeric = (loss_and_grad(&l, &a, &dets, &errors, 16, 1).0 - loss_and_grad(&l, &b, &dets, &errors, 16, 1).0) / 2e-2;
            assert!((numeric - grad[i] as f64).abs() < 2e-3 + 2e-2 * numeric.abs(), "{i}: {numeric} vs {}", grad[i]);
        }
    }

    #[test]
    fn threads_change_nothing() {
        let d = dem();
        let l = Layout::new(&d, None).unwrap();
        let net = Net::new(&[l.num_inputs(), 5, l.num_kinds()], 3, 2).unwrap();
        let (dets, _, errors) = DemSampler::new(&d).sample_seeded(4, 0, 50, 1, true);
        let one = loss_and_grad(&l, &net, &dets, &errors, 50, 1);
        for threads in [2, 3, 7] {
            let many = loss_and_grad(&l, &net, &dets, &errors, 50, threads);
            assert_eq!(one.0.to_bits(), many.0.to_bits());
            assert_eq!(one.1, many.1);
        }
    }

    #[test]
    fn training_learns_the_isolated_faults() {
        let d = dem();
        let l = Layout::new(&d, None).unwrap();
        let mut net = Net::new(&[l.num_inputs(), 8, l.num_kinds()], 3, 1).unwrap();
        let mut tr = Trainer::new(Optimizer::Adam, net.params.len());
        let s = DemSampler::new(&d);
        let losses = train(&l, &mut net, &mut tr, &s, 300, 64, 1e-2, 10, 3, 2);
        assert!(losses[290..].iter().sum::<f64>() < 0.7 * losses[..10].iter().sum::<f64>());
        // D2 alone and D2 D3 together: the network applies D2 L0 and D2 D3.
        let (left, obs, applied) = predecode(&l, &net, &[0b0100, 0b1100], 2, 0.5, 1);
        assert_eq!((left, obs, applied), (vec![0, 0], vec![1, 0], vec![1, 1]));
    }
}

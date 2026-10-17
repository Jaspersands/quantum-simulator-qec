//! The lane program on a GPU, through wgpu (Metal, Vulkan, DX12). `sample.wgsl` interprets the
//! same `Vec<u32>` the CPU executor in `lanes` runs, one invocation per word of 32 shots, and
//! gives the same bits: the CPU is the reference, the GPU the throughput.
//!
//! One device is opened on first use and kept. A sample is cut into dispatches of as many words
//! as a memory budget allows (frames, the ring of records, detectors, observables and, for a
//! measurement sampler, every record live per word), each dispatch read back and packed into
//! b8 rows on the CPU as the lane sampler packs its own.

use std::sync::{Mutex, OnceLock};

use crate::lanes::{word_seed, write_b8, LaneProgram, LANES};

const SHADER: &str = include_str!("sample.wgsl");
const BP_SHADER: &str = include_str!("bp.wgsl");
/// Bytes of device memory a dispatch may use for its words' buffers.
const BUDGET: usize = 256 << 20;
const WORKGROUP: u32 = 64;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    bp_pipeline: wgpu::ComputePipeline,
    bp_layout: wgpu::BindGroupLayout,
    name: String,
    max_buffer: usize,
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

fn make_pipeline(device: &wgpu::Device, label: &str, source: &str, entries: &[wgpu::BindGroupLayoutEntry]) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(source.into()) });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipeline, layout)
}

static GPU: OnceLock<Result<Mutex<Gpu>, String>> = OnceLock::new();

fn open() -> Result<Gpu, String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        ..Default::default()
    }))
    .map_err(|e| format!("no GPU adapter: {e}"))?;
    let info = adapter.get_info();
    let limits = adapter.limits();
    let mut required = wgpu::Limits::downlevel_defaults().using_resolution(limits.clone());
    required.max_storage_buffer_binding_size = limits.max_storage_buffer_binding_size;
    required.max_buffer_size = limits.max_buffer_size;
    required.max_storage_buffers_per_shader_stage = limits.max_storage_buffers_per_shader_stage.max(8);
    required.max_compute_workgroups_per_dimension = limits.max_compute_workgroups_per_dimension;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("stabilizer-qec"),
        required_features: wgpu::Features::empty(),
        required_limits: required,
        ..Default::default()
    }))
    .map_err(|e| format!("the GPU adapter {} refused a device: {e}", info.name))?;
    let (pipeline, layout) = make_pipeline(
        &device,
        "lane sampler",
        SHADER,
        &[storage_entry(0, true), uniform_entry(1), storage_entry(2, true), storage_entry(3, false), storage_entry(4, false), storage_entry(5, false), storage_entry(6, false), storage_entry(7, false)],
    );
    let (bp_pipeline, bp_layout) = make_pipeline(
        &device,
        "min-sum BP",
        BP_SHADER,
        &[uniform_entry(0), storage_entry(1, true), storage_entry(2, true), storage_entry(3, true), storage_entry(4, false), storage_entry(5, false), storage_entry(6, false)],
    );
    let max_buffer = (limits.max_storage_buffer_binding_size as usize).min(limits.max_buffer_size as usize);
    Ok(Gpu { device, queue, pipeline, layout, bp_pipeline, bp_layout, name: format!("{} ({:?})", info.name, info.backend), max_buffer })
}

fn gpu() -> Result<&'static Mutex<Gpu>, String> {
    GPU.get_or_init(|| open().map(Mutex::new)).as_ref().map_err(Clone::clone)
}

/// Opens the device if it is not open; an error names why there is none.
pub fn device_ready() -> Result<(), String> {
    gpu().map(|_| ())
}

/// The device's name, once it is open.
pub fn device_name() -> Result<String, String> {
    Ok(gpu()?.lock().map_err(|_| "the GPU lock is poisoned")?.name.clone())
}

fn as_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

fn from_bytes(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}


/// How many words or shots a dispatch takes: a first small one, then as many as ran in about
/// `TARGET_SECONDS` last time, within the memory budget. A dispatch that runs for seconds trips
/// the operating system's GPU watchdog (macOS kills the command buffer after a few seconds and
/// the rest of the results come back zero), so the work is paced, not just the memory.
struct Pacer {
    max: usize,
    next: usize,
}

const TARGET_SECONDS: f64 = 0.2;

impl Pacer {
    fn new(max: usize) -> Pacer {
        Pacer { max: max.max(1), next: 512.min(max.max(1)) }
    }

    fn take(&self, remaining: usize) -> usize {
        self.next.min(remaining).max(1)
    }

    fn observe(&mut self, n: usize, seconds: f64) {
        let rate = n as f64 / seconds.max(1e-4);
        self.next = ((rate * TARGET_SECONDS) as usize).clamp(64, self.max);
    }
}

struct Readback {
    detectors: Vec<u32>,
    observables: Vec<u32>,
    measurements: Vec<u32>,
}

impl Gpu {
    fn buffer(&self, label: &str, size: usize, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (size.max(4) as u64).next_multiple_of(4), usage, mapped_at_creation: false })
    }

    fn upload(&self, label: &str, words: &[u32], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let b = self.buffer(label, words.len() * 4, usage | wgpu::BufferUsages::COPY_DST);
        if !words.is_empty() {
            self.queue.write_buffer(&b, 0, &as_bytes(words));
        }
        b
    }

    fn read(&self, encoder_src: &wgpu::Buffer, size: usize) -> Result<Vec<u32>, String> {
        if size == 0 {
            return Ok(Vec::new());
        }
        let staging = self.buffer("readback", size, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        encoder.copy_buffer_to_buffer(encoder_src, 0, &staging, 0, size as u64);
        self.queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| format!("the GPU did not finish: {e:?}"))?;
        rx.recv().map_err(|_| "the GPU readback was dropped")?.map_err(|e| format!("mapping the GPU readback failed: {e}"))?;
        let out = {
            let view = slice.get_mapped_range().map_err(|e| format!("reading the GPU readback failed: {e:?}"))?;
            from_bytes(&view)
        };
        staging.unmap();
        Ok(out)
    }

    /// Words `first..first + n` of `prog`'s stream in one dispatch.
    fn run(&self, prog: &LaneProgram, code: &wgpu::Buffer, seed: u64, first: u64, n: usize, keep: bool) -> Result<Readback, String> {
        let nq = prog.num_qubits.max(1);
        let ring = prog.ring_mask as usize + 1;
        let seeds: Vec<u32> = (0..n).map(|w| word_seed(seed, first + w as u64)).collect();
        let params = [n as u32, prog.num_qubits as u32, prog.ring_mask, prog.num_detectors as u32, prog.num_observables as u32, prog.num_measurements as u32, u32::from(keep), prog.code.len() as u32];
        let params_buf = self.upload("params", &params, wgpu::BufferUsages::UNIFORM);
        let seeds_buf = self.upload("seeds", &seeds, wgpu::BufferUsages::STORAGE);
        let rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let frames = self.buffer("frames", n * 2 * nq * 4, rw);
        let ring_buf = self.buffer("ring", n * ring * 4, rw);
        let dets = self.buffer("detectors", n * prog.num_detectors * 4, rw);
        let obs = self.buffer("observables", n * prog.num_observables * 4, rw);
        let meas = self.buffer("measurements", if keep { n * prog.num_measurements * 4 } else { 4 }, rw);
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lane sampler"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: code.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: seeds_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: frames.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: ring_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: dets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: obs.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: meas.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("lane sampler") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("lane sampler"), timestamp_writes: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP), 1, 1);
        }
        self.queue.submit([encoder.finish()]);
        Ok(Readback {
            detectors: self.read(&dets, n * prog.num_detectors * 4)?,
            observables: self.read(&obs, n * prog.num_observables * 4)?,
            measurements: if keep { self.read(&meas, n * prog.num_measurements * 4)? } else { Vec::new() },
        })
    }

    /// How many words a dispatch may hold within the budget and the device's buffer limits.
    fn words_per_dispatch(&self, prog: &LaneProgram, keep: bool) -> Result<usize, String> {
        let per_word = 4 * (2 * prog.num_qubits.max(1) + prog.ring_mask as usize + 1 + prog.num_detectors + prog.num_observables + if keep { prog.num_measurements } else { 0 });
        let largest = 4 * [2 * prog.num_qubits.max(1), prog.ring_mask as usize + 1, prog.num_detectors, prog.num_observables, if keep { prog.num_measurements } else { 0 }].into_iter().max().unwrap_or(1).max(1);
        let by_budget = BUDGET / per_word.max(1);
        let by_buffer = self.max_buffer / largest;
        let by_dispatch = (65_535 * WORKGROUP) as usize;
        let n = by_budget.min(by_buffer).min(by_dispatch);
        if n == 0 {
            return Err(format!("the circuit's {per_word} bytes per word of 32 shots exceed the GPU's buffers"));
        }
        Ok(n)
    }
}

/// `shots` shots from word `first` on as b8 rows of detectors and of observables, as
/// `LaneProgram::sample_seeded` gives them and bit for bit the same.
pub fn sample_seeded(prog: &LaneProgram, seed: u64, first: u64, shots: usize) -> Result<(Vec<u8>, Vec<u8>), String> {
    let g = gpu()?.lock().map_err(|_| "the GPU lock is poisoned")?;
    let (ds, os) = (prog.num_detectors.div_ceil(8), prog.num_observables.div_ceil(8));
    let mut dets = Vec::with_capacity(shots * ds);
    let mut obs = Vec::with_capacity(shots * os);
    let words = shots.div_ceil(LANES);
    let mut pacer = Pacer::new(g.words_per_dispatch(prog, false)?);
    let code = g.upload("program", &prog.code, wgpu::BufferUsages::STORAGE);
    let mut done = 0usize;
    while done < words {
        let n = pacer.take(words - done);
        let t0 = std::time::Instant::now();
        let back = g.run(prog, &code, seed, first + done as u64, n, false)?;
        pacer.observe(n, t0.elapsed().as_secs_f64());
        for w in 0..n {
            let lanes = (shots - (done + w) * LANES).min(LANES);
            write_b8(&back.detectors[w * prog.num_detectors..(w + 1) * prog.num_detectors], lanes, ds, &mut dets);
            write_b8(&back.observables[w * prog.num_observables..(w + 1) * prog.num_observables], lanes, os, &mut obs);
        }
        done += n;
    }
    Ok((dets, obs))
}

/// `shots` measurement records as b8 rows, the `reference` with each shot's flips, as
/// `LaneProgram::sample_measurements_seeded` gives them.
pub fn sample_measurements_seeded(prog: &LaneProgram, seed: u64, first: u64, shots: usize, reference: &[bool]) -> Result<Vec<u8>, String> {
    let g = gpu()?.lock().map_err(|_| "the GPU lock is poisoned")?;
    let stride = prog.num_measurements.div_ceil(8);
    let mut base = vec![0u8; stride];
    for (k, &b) in reference.iter().enumerate() {
        if b {
            base[k / 8] |= 1 << (k % 8);
        }
    }
    let mut rows = Vec::with_capacity(shots * stride);
    let words = shots.div_ceil(LANES);
    let mut pacer = Pacer::new(g.words_per_dispatch(prog, true)?);
    let code = g.upload("program", &prog.code, wgpu::BufferUsages::STORAGE);
    let mut done = 0usize;
    while done < words {
        let n = pacer.take(words - done);
        let t0 = std::time::Instant::now();
        let back = g.run(prog, &code, seed, first + done as u64, n, true)?;
        pacer.observe(n, t0.elapsed().as_secs_f64());
        for w in 0..n {
            let lanes = (shots - (done + w) * LANES).min(LANES);
            let start = rows.len();
            write_b8(&back.measurements[w * prog.num_measurements..(w + 1) * prog.num_measurements], lanes, stride, &mut rows);
            for lane in 0..lanes {
                for (k, &b) in base.iter().enumerate() {
                    rows[start + lane * stride + k] ^= b;
                }
            }
        }
        done += n;
    }
    Ok(rows)
}

/// Min-sum BP in f32 on the GPU: `syndromes` one byte per check per shot, back to back; the
/// outcome as `Bp32::decode_batch` gives it and bit for bit the same.
pub fn bp_decode_batch(bp: &crate::bp32::Bp32, syndromes: &[u8], shots: usize, scale: f32, max_iter: usize) -> Result<crate::bp32::Bp32Outcome, String> {
    use crate::bp32::{scales, Bp32Outcome};
    let g = gpu()?.lock().map_err(|_| "the GPU lock is poisoned")?;
    let (nc, nv, ne) = (bp.num_checks, bp.num_vars, bp.num_edges());
    if syndromes.len() != shots * nc {
        return Err(format!("{} syndrome bytes for {shots} shots of {nc} checks", syndromes.len()));
    }
    let stride = nv.div_ceil(8);
    let mut out = Bp32Outcome { hard: Vec::with_capacity(shots * stride), converged: Vec::with_capacity(shots), iterations: Vec::with_capacity(shots) };
    if shots == 0 {
        return Ok(out);
    }
    // The graph and the floats, once.
    let mut graph = Vec::with_capacity(bp.var_start.len() + 3 * ne + bp.chk_start.len());
    let off_var_start = graph.len() as u32;
    graph.extend_from_slice(&bp.var_start);
    let off_edge_check = graph.len() as u32;
    graph.extend_from_slice(&bp.edge_check);
    let off_edge_var = graph.len() as u32;
    graph.extend_from_slice(&bp.edge_var);
    let off_chk_start = graph.len() as u32;
    graph.extend_from_slice(&bp.chk_start);
    let off_chk_edges = graph.len() as u32;
    graph.extend_from_slice(&bp.chk_edges);
    let graph_buf = g.upload("bp graph", &graph, wgpu::BufferUsages::STORAGE);
    let mut floats: Vec<u32> = bp.prior_llr.iter().map(|f| f.to_bits()).collect();
    let off_scales = floats.len() as u32;
    floats.extend(scales(scale, max_iter).iter().map(|f| f.to_bits()));
    let floats_buf = g.upload("bp floats", &floats, wgpu::BufferUsages::STORAGE);
    // Per shot: 2·ne messages, nv posteriors, nv + nc + 2 words of work, nc of syndrome.
    let per_shot = 4 * (2 * ne + nv + nv + nc + 2 + nc);
    let largest = 4 * (2 * ne).max(nv + nc + 2).max(nc).max(nv).max(1);
    let mut pacer = Pacer::new((BUDGET / per_shot.max(1)).min(g.max_buffer / largest).min((65_535 * WORKGROUP) as usize));
    let mut done = 0usize;
    while done < shots {
        let n = pacer.take(shots - done);
        let t0 = std::time::Instant::now();
        let params = [n as u32, nc as u32, nv as u32, ne as u32, max_iter as u32, off_var_start, off_edge_check, off_edge_var, off_chk_start, off_chk_edges, off_scales, 0];
        let params_buf = g.upload("bp params", &params, wgpu::BufferUsages::UNIFORM);
        let synd: Vec<u32> = syndromes[done * nc..(done + n) * nc].iter().map(|&s| u32::from(s != 0)).collect();
        let synd_buf = g.upload("bp syndromes", &synd, wgpu::BufferUsages::STORAGE);
        let rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let messages = g.buffer("bp messages", n * 2 * ne * 4, rw);
        let llr = g.buffer("bp llr", n * nv * 4, rw);
        let work = g.buffer("bp work", n * (nv + nc + 2) * 4, rw);
        let bind = g.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("min-sum BP"),
            layout: &g.bp_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: graph_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: floats_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: synd_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: messages.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: llr.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: work.as_entire_binding() },
            ],
        });
        let mut encoder = g.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("min-sum BP") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("min-sum BP"), timestamp_writes: None });
            pass.set_pipeline(&g.bp_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP), 1, 1);
        }
        g.queue.submit([encoder.finish()]);
        let back = g.read(&work, n * (nv + nc + 2) * 4)?;
        pacer.observe(n, t0.elapsed().as_secs_f64());
        for s in 0..n {
            let base = s * (nv + nc + 2);
            let start = out.hard.len();
            out.hard.resize(start + stride, 0);
            for v in 0..nv {
                if back[base + v] != 0 {
                    out.hard[start + v / 8] |= 1 << (v % 8);
                }
            }
            out.converged.push(u8::from(back[base + nv + nc] != 0));
            out.iterations.push(back[base + nv + nc + 1]);
        }
        done += n;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;

    #[test]
    fn gpu_bp_equals_cpu_f32() {
        if !available() {
            return;
        }
        use crate::bp::Bp;
        use crate::bp32::Bp32;
        // A d = 5 surface memory's decomposed-free model as a check matrix: detectors × faults.
        let noise = crate::generated::Noise {
            after_clifford_depolarization: 0.004,
            before_round_data_depolarization: 0.0,
            before_measure_flip_probability: 0.004,
            after_reset_flip_probability: 0.0,
            after_reset_leakage: 0.0,
            after_clifford_leakage_and_relaxation: 0.0,
        };
        let c = Circuit::parse(&crate::generated::generate("surface_code:rotated_memory_z", 5, 5, &noise).unwrap()).unwrap();
        let dem = crate::dem::Dem::from_circuit_undecomposed(&c).unwrap();
        let m = dem.num_detectors;
        let mut ids: std::collections::HashMap<Vec<u32>, usize> = std::collections::HashMap::new();
        let (mut columns, mut priors): (Vec<Vec<u32>>, Vec<f64>) = (Vec::new(), Vec::new());
        for mech in &dem.mechanisms {
            let id = *ids.entry(mech.detectors.clone()).or_insert_with(|| {
                columns.push(mech.detectors.clone());
                priors.push(0.0);
                columns.len() - 1
            });
            let q = priors[id];
            priors[id] = q * (1.0 - mech.p) + mech.p * (1.0 - q);
        }
        let bp = Bp::new(m, &columns, &priors).unwrap();
        let bp32 = Bp32::from_bp(&bp);
        let prog = crate::lanes::LaneProgram::new(&c).unwrap();
        let (dets, _) = prog.sample_seeded(1, 0, 3000, 4);
        let stride = prog.num_detectors.div_ceil(8);
        let dets = &dets;
        let synd: Vec<u8> = (0..3000).flat_map(|s| (0..m).map(move |c| (dets[s * stride + c / 8] >> (c % 8)) & 1)).collect();
        for (scale, iters) in [(0.0f32, 20usize), (0.625, 30), (1.0, 5)] {
            let cpu = bp32.decode_batch(&synd, 3000, scale, iters, 4);
            let gpu = bp_decode_batch(&bp32, &synd, 3000, scale, iters).unwrap();
            assert_eq!(cpu.hard, gpu.hard, "scale {scale}");
            assert_eq!(cpu.converged, gpu.converged, "scale {scale}");
            assert_eq!(cpu.iterations, gpu.iterations, "scale {scale}");
        }
    }

    fn available() -> bool {
        device_ready().is_ok()
    }

    #[test]
    fn gpu_equals_cpu_lanes() {
        if !available() {
            eprintln!("no GPU device; skipped");
            return;
        }
        let noise = crate::generated::Noise {
            after_clifford_depolarization: 0.01,
            before_round_data_depolarization: 0.005,
            before_measure_flip_probability: 0.01,
            after_reset_flip_probability: 0.002,
            after_reset_leakage: 0.0,
            after_clifford_leakage_and_relaxation: 0.0,
        };
        for (task, d, r) in [("surface_code:rotated_memory_z", 3, 3), ("surface_code:rotated_memory_x", 5, 7), ("color_code:memory_xyz", 3, 2), ("repetition_code:memory", 9, 9)] {
            let c = Circuit::parse(&crate::generated::generate(task, d, r, &noise).unwrap()).unwrap();
            let prog = LaneProgram::new(&c).unwrap();
            for shots in [1usize, 31, 32, 33, 700] {
                let cpu = prog.sample_seeded(42, 3, shots, 4);
                let gpu = sample_seeded(&prog, 42, 3, shots).unwrap();
                assert_eq!(cpu, gpu, "{task} d={d} shots={shots}");
            }
            let cpu = prog.sample_measurements_seeded(42, 0, 100, 2, &[]);
            let gpu = sample_measurements_seeded(&prog, 42, 0, 100, &[]).unwrap();
            assert_eq!(cpu, gpu, "{task} measurements");
        }
    }

    #[test]
    fn gpu_runs_every_channel() {
        if !available() {
            return;
        }
        let text = "R 0 1 2 3\nH 0\nCX 0 1\nCZ 1 2\nS 2\nX_ERROR(0.1) 0\nY_ERROR(0.1) 1\nZ_ERROR(0.1) 2\nDEPOLARIZE1(0.2) 0 1\nDEPOLARIZE2(0.2) 0 1\nPAULI_CHANNEL_1(0.05, 0.1, 0.15) 2\nPAULI_CHANNEL_2(0.01, 0.02, 0.03, 0.04, 0.01, 0.02, 0.03, 0.04, 0.01, 0.02, 0.03, 0.04, 0.01, 0.02, 0.03) 2 3\nE(0.3) X0 Z1\nELSE_CORRELATED_ERROR(0.5) Y2\nHERALDED_ERASE(0.2) 3\nMPAD 0 1\nM(0.05) 0 1\nMR 2\nMX 3\nDETECTOR rec[-1]\nDETECTOR rec[-2] rec[-3]\nREPEAT 4 {\n    X_ERROR(0.1) 0\n    CX 1 0\n    M 0\n    DETECTOR rec[-1] rec[-2]\n    CX rec[-1] 1\n    REPEAT 2 {\n        Z_ERROR(0.3) 1\n        MX 1\n        DETECTOR rec[-1]\n    }\n}\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-3]\nOBSERVABLE_INCLUDE(1) Z0 X1";
        let c = Circuit::parse(text).unwrap();
        let prog = LaneProgram::new(&c).unwrap();
        assert_eq!(prog.sample_seeded(9, 0, 5000, 4), sample_seeded(&prog, 9, 0, 5000).unwrap());
        assert_eq!(prog.sample_measurements_seeded(9, 7, 333, 4, &[true, false, true]), sample_measurements_seeded(&prog, 9, 7, 333, &[true, false, true]).unwrap());
    }
}

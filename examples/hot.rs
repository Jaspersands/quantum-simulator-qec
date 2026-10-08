//! The jobs `tools/bench.py` times against Stim and PyMatching, run natively for a profiler.
//!
//! The recording machine's default Rust toolchain is x86_64 and runs under Rosetta; always pass
//! the native target there:
//!
//!     cargo run --release --no-default-features --target aarch64-apple-darwin --example hot -- dem 20
//!     (while it runs, in another shell: sample <pid> 5 -file $SCRATCH/dem.txt)
//!
//! Jobs: `dem`, `graphlike`, `sample`, `m2d`, `match`, `corr`; then the number of repetitions,
//! and optionally the distance (default 11, the benchmarks'). Each repetition's time is printed,
//! and the best.
use std::time::Instant;

use stabilizer_qec::batch_sampler::BatchSampler;
use stabilizer_qec::{BitTable, Circuit, DemOptions, GeneratedNoise, Matching};

fn memory(d: u32) -> Circuit {
    let p = 0.001;
    let noise = GeneratedNoise::new().after_clifford_depolarization(p).before_round_data_depolarization(p).before_measure_flip_probability(p).after_reset_flip_probability(p);
    Circuit::generated("surface_code:rotated_memory_z", d, d as u64, &noise).unwrap()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let job = a.get(1).map(String::as_str).unwrap_or("dem");
    let reps: usize = a.get(2).map_or(10, |s| s.parse().unwrap());
    let d: u32 = a.get(3).map_or(11, |s| s.parse().unwrap());
    let shots = 20_000;
    let c = memory(d);
    let decomposed = c.detector_error_model(&DemOptions::new().decompose_errors(true)).unwrap();
    let dets = c.detector_sampler(1).unwrap().sample(shots, 1).detectors;
    let engine = stabilizer_qec::circuit::Circuit::parse(&c.to_string()).unwrap();
    let measurements = {
        let rows = BatchSampler::new(&engine).unwrap().sample_measurements_seeded(2, 0, shots, 1, &[]);
        BitTable::from_packed(shots, c.num_measurements(), rows).unwrap()
    };
    let converter = c.measurement_converter().unwrap();
    let plain = Matching::new(&decomposed).unwrap();
    let correlated = Matching::with_correlations(&decomposed).unwrap();
    let mut sampler = c.detector_sampler(3).unwrap();
    let mut run = || match job {
        "dem" => drop(c.detector_error_model(&DemOptions::new().decompose_errors(true)).unwrap()),
        "graphlike" => drop(decomposed.shortest_graphlike_error(true).unwrap()),
        "sample" => drop(sampler.sample(shots, 1)),
        "m2d" => drop(converter.convert(&measurements, None).unwrap()),
        "match" => drop(plain.decode_batch(&dets, 1).unwrap()),
        "corr" => drop(correlated.decode_batch(&dets, 1).unwrap()),
        _ => panic!("unknown job {job}: dem, graphlike, sample, m2d, match or corr"),
    };
    run();
    let mut best = f64::INFINITY;
    for k in 0..reps {
        let t = Instant::now();
        run();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        best = best.min(ms);
        eprintln!("{job} d={d} rep {k}: {ms:.3} ms");
    }
    println!("{job} d={d}: best {best:.3} ms");
}

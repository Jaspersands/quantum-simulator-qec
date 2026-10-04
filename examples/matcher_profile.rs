//! The sparse matcher, run natively for a profiler or a timing table.
//!
//! The recording machine's default Rust toolchain is x86_64 and runs under
//! Rosetta, which roughly halves the speed; always pass the native target:
//!
//!     cargo run --release --no-default-features --target aarch64-apple-darwin \
//!         --example matcher_profile -- profile 11 0.006 plain 12
//!     (then, in another shell: sample <pid> 6 -f profile.txt)
//!     cargo run ... --example matcher_profile -- timing data/matcher/native.json
use std::fmt::Write as _;

use stabilizer_qec::circuit::Basis;
use stabilizer_qec::dem::Dem;
use stabilizer_qec::dem_decoder::DemDecoder;
use stabilizer_qec::frame_sampler::FrameSampler;
use stabilizer_qec::memory::{generate, CodeKind, NoiseModel};
use stabilizer_qec::sparse::Scratch;
use stabilizer_qec::surface_code::Xorshift;

fn shots(d: usize, p: f64, n: usize) -> (DemDecoder, Vec<Vec<u32>>) {
    let c = generate(CodeKind::Rotated, d, d, NoiseModel::Sd6 { p }, Basis::Z).unwrap();
    let dec = DemDecoder::new(&Dem::from_circuit(&c).unwrap()).unwrap();
    let sampler = FrameSampler::new(&c).unwrap();
    let mut rng = Xorshift::new(3);
    let shots = (0..n)
        .map(|_| {
            sampler
                .sample(&mut rng)
                .detectors
                .iter()
                .enumerate()
                .filter(|x| *x.1)
                .map(|x| x.0 as u32)
                .collect()
        })
        .collect();
    (dec, shots)
}

/// Mean microseconds a shot over the best of three passes. Each pass repeats
/// the shots for at least a quarter second: a pass of a millisecond can end
/// before the scheduler has moved the process onto a performance core.
fn time_us(shots: &[Vec<u32>], mut f: impl FnMut(&[u32])) -> f64 {
    (0..3)
        .map(|_| {
            let t = std::time::Instant::now();
            let mut n = 0usize;
            while n == 0 || t.elapsed().as_secs_f64() < 0.25 {
                for s in shots {
                    f(s);
                }
                n += shots.len();
            }
            t.elapsed().as_secs_f64() * 1e6 / n as f64
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("profile") => {
            let (d, p): (usize, f64) = (a[2].parse().unwrap(), a[3].parse().unwrap());
            let corr = a[4] == "corr";
            let secs: f64 = a[5].parse().unwrap();
            let (dec, shots) = shots(d, p, 4000);
            let mut s = Scratch::new(dec.graph());
            let t = std::time::Instant::now();
            let mut n = 0u64;
            while t.elapsed().as_secs_f64() < secs {
                for sh in &shots {
                    if corr {
                        dec.graph()
                            .decode_correlated(dec.correlations(), &mut s, sh)
                            .unwrap();
                    } else {
                        dec.graph().decode(&mut s, sh).unwrap();
                    }
                    n += 1;
                }
            }
            println!("{:.2} us/shot", t.elapsed().as_secs_f64() * 1e6 / n as f64);
        }
        Some("timing") => {
            let mut json = String::from("{\"points\": [\n");
            let mut first = true;
            for d in [3usize, 5, 7, 9] {
                for p in [0.003, 0.006] {
                    let (dec, shots) = shots(d, p, 2000);
                    let mut s = Scratch::new(dec.graph());
                    let sparse = time_us(&shots, |sh| {
                        dec.graph().decode(&mut s, sh).unwrap();
                    });
                    let corr = time_us(&shots, |sh| {
                        dec.graph()
                            .decode_correlated(dec.correlations(), &mut s, sh)
                            .unwrap();
                    });
                    let dense = time_us(&shots[..300], |sh| {
                        let _ = dec.decode_dense(sh);
                    });
                    println!("d = {d}, p = {p}: sparse {sparse:.2} us, correlated {corr:.2} us, dense {dense:.1} us");
                    let _ = write!(json, "{} {{\"d\": {d}, \"p\": {p}, \"sparse_us\": {sparse}, \"correlated_us\": {corr}, \"dense_us\": {dense}}}",
                        if first { "" } else { ",\n" });
                    first = false;
                }
            }
            json.push_str(&format!(
                "\n], \"target\": \"{}\"}}\n",
                std::env::consts::ARCH
            ));
            std::fs::write(&a[2], json).unwrap();
        }
        _ => eprintln!("usage: matcher_profile profile D P plain|corr SECONDS | timing OUT.json"),
    }
}

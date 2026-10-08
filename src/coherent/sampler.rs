//! The coherent sampler: shots of the twirled circuit, as the frame sampler draws them, each
//! with its weight. Weighted, the shots are distributed as the coherent circuit's; the estimate
//! of any rate is Σ w·[event] / Σ w.

use std::collections::HashMap;

use super::kernel::{CoherentOptions, Kernel};
use super::shot::Stream;
use super::{GaugeSolver, Program};
use crate::circuit::Circuit;
use crate::surface_code::Xorshift;

pub struct CoherentSampler {
    pub program: Program,
    pub kernel: Kernel,
    solver: GaugeSolver,
}

/// Weighted shots: detection events and observable flips as b8 rows, and each shot's weight.
pub struct CoherentBatch {
    pub detectors: Vec<u8>,
    pub observables: Vec<u8>,
    pub weights: Vec<f64>,
}

impl CoherentSampler {
    pub fn new(circuit: &Circuit, options: CoherentOptions) -> Result<CoherentSampler, String> {
        let mut program = Program::new(circuit)?;
        let solver = GaugeSolver::new(&program);
        let kernel = Kernel::new(&program, &solver, options);
        program.set_merge(&kernel.merge_groups());
        Ok(CoherentSampler { program, kernel, solver })
    }

    /// Shots `first..first + shots`, shot s from its own stream (`batch_seed(seed, s)`), so
    /// threads never change them.
    pub fn sample_seeded(&self, seed: u64, first: u64, shots: usize, threads: usize) -> CoherentBatch {
        let p = &self.program;
        let (nd, no) = (p.num_detectors(), p.num_observables());
        let (dw, ow) = (nd.div_ceil(8), no.div_ceil(8));
        let rows = crate::parallel::parallel(shots, threads, |range| {
            let mut cache = HashMap::new();
            range
                .map(|s| {
                    let mut rng = Xorshift::new(crate::batch_sampler::batch_seed(seed, first + s as u64));
                    let shot = p.run_shot(&mut Stream(&mut rng));
                    let w = self.kernel.weight(p, &shot, &self.solver, &mut cache);
                    (p.outcome(&shot.records), w)
                })
                .collect()
        });
        let mut batch = CoherentBatch { detectors: vec![0u8; shots * dw], observables: vec![0u8; shots * ow], weights: Vec::with_capacity(shots) };
        for (s, ((dets, obs), w)) in rows.into_iter().enumerate() {
            for (k, &b) in dets.iter().enumerate() {
                if b {
                    batch.detectors[s * dw + k / 8] |= 1 << (k % 8);
                }
            }
            for k in 0..no {
                if obs >> k & 1 == 1 {
                    batch.observables[s * ow + k / 8] |= 1 << (k % 8);
                }
            }
            batch.weights.push(w);
        }
        batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_sampler::FrameSampler;

    #[test]
    fn without_rotations_the_shots_are_the_frame_samplers() {
        let text = crate::fixtures::REP3.replace("0.01", "0.05");
        let tagged = text.replace("TICK", "TICK\nI_ERROR[R_X(theta=0)] 0 1 2");
        let c = Circuit::parse(&tagged).unwrap();
        let p = Program::new(&c).unwrap();
        let f = FrameSampler::new(&c).unwrap();
        for seed in 0..200 {
            let (mut a, mut b) = (Xorshift::new(seed), Xorshift::new(seed));
            let shot = p.run_shot(&mut Stream(&mut a));
            let theirs = f.sample(&mut b);
            assert_eq!(p.outcome(&shot.records), (theirs.detectors, theirs.observables));
        }
    }

    #[test]
    fn threads_do_not_change_the_shots() {
        let c = Circuit::parse("R 0 1\nH 0\nI_ERROR[R_X(theta=0.3)] 1\nCX 0 1\nM 0 1\nDETECTOR rec[-1] rec[-2]").unwrap();
        let s = CoherentSampler::new(&c, CoherentOptions::default()).unwrap();
        let (a, b) = (s.sample_seeded(5, 0, 300, 1), s.sample_seeded(5, 0, 300, 4));
        assert_eq!(a.detectors, b.detectors);
        assert_eq!(a.weights, b.weights);
    }
}

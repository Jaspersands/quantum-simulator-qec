//! Sampling a detector error model directly (Stim's `DetectorErrorModel.compile_sampler`):
//! each fault fires independently with its probability, and the detectors and observables it
//! flips are flipped. Shots come 64 at a time, each batch from a stream seeded by the sampler's
//! seed and the batch's index, so a seed gives the same shots on any number of threads; the
//! faults that fired can be returned, and recorded faults replayed.

use crate::batch_sampler::{batch_seed, bernoulli, Noise};
use crate::dem::Dem;
use crate::surface_code::Xorshift;

pub struct DemSampler {
    faults: Vec<(Noise, Vec<u32>, u64)>,
    pub num_detectors: usize,
    pub num_observables: usize,
}

/// Packed rows (b8): detectors, observables, and (when asked) the faults that fired.
pub type Rows = (Vec<u8>, Vec<u8>, Vec<u8>);

/// Bit `k` of 64-shot words `words` as `shots` packed rows of `words.len()` bits.
fn write_rows(words: &[u64], shots: usize, out: &mut Vec<u8>) {
    let stride = words.len().div_ceil(8);
    for k in 0..shots {
        let start = out.len();
        out.resize(start + stride, 0);
        for (j, w) in words.iter().enumerate() {
            if w >> k & 1 == 1 {
                out[start + j / 8] |= 1 << (j % 8);
            }
        }
    }
}

impl DemSampler {
    pub fn new(dem: &Dem) -> DemSampler {
        let faults = dem.mechanisms.iter().map(|m| (Noise::new(m.p), m.detectors.clone(), m.observables)).collect();
        DemSampler { faults, num_detectors: dem.num_detectors, num_observables: dem.num_observables }
    }

    pub fn num_errors(&self) -> usize {
        self.faults.len()
    }

    /// The detectors and observables flipped when the faults of `fired` (one word per fault,
    /// bit k for shot k) fire.
    fn apply(&self, fired: &[u64]) -> (Vec<u64>, Vec<u64>) {
        let mut dets = vec![0u64; self.num_detectors];
        let mut obs = vec![0u64; self.num_observables];
        for ((_, ds, os), &w) in self.faults.iter().zip(fired) {
            if w == 0 {
                continue;
            }
            for &d in ds {
                dets[d as usize] ^= w;
            }
            for (k, o) in obs.iter_mut().enumerate() {
                if os >> k & 1 == 1 {
                    *o ^= w;
                }
            }
        }
        (dets, obs)
    }

    /// `shots` shots, batch `first + b` from its own stream; `errors` also returns the faults
    /// that fired.
    pub fn sample_seeded(&self, seed: u64, first: u64, shots: usize, threads: usize, errors: bool) -> Rows {
        let parts = crate::parallel::parallel(shots.div_ceil(64), threads, |range| {
            let (mut d, mut o, mut e) = (Vec::new(), Vec::new(), Vec::new());
            for b in range {
                let mut rng = Xorshift::new(batch_seed(seed, first + b as u64));
                let fired: Vec<u64> = self.faults.iter().map(|(n, _, _)| bernoulli(&mut rng, *n)).collect();
                let (dets, obs) = self.apply(&fired);
                let here = (shots - b * 64).min(64);
                write_rows(&dets, here, &mut d);
                write_rows(&obs, here, &mut o);
                if errors {
                    write_rows(&fired, here, &mut e);
                }
            }
            vec![(d, o, e)]
        });
        let (mut d, mut o, mut e) = (Vec::new(), Vec::new(), Vec::new());
        for (a, b, c) in parts {
            d.extend(a);
            o.extend(b);
            e.extend(c);
        }
        (d, o, e)
    }

    /// The detection events and observable flips of recorded faults (`errors`: `shots` packed
    /// rows of `num_errors` bits).
    pub fn replay(&self, errors: &[u8], shots: usize) -> Result<(Vec<u8>, Vec<u8>), String> {
        let stride = self.faults.len().div_ceil(8);
        if errors.len() != stride * shots {
            return Err(format!("recorded errors must be {shots} rows of {} bits", self.faults.len()));
        }
        let (mut d, mut o) = (Vec::new(), Vec::new());
        for b in 0..shots.div_ceil(64) {
            let here = (shots - b * 64).min(64);
            let mut fired = vec![0u64; self.faults.len()];
            for k in 0..here {
                let row = &errors[(b * 64 + k) * stride..(b * 64 + k + 1) * stride];
                for (j, w) in fired.iter_mut().enumerate() {
                    if row[j / 8] >> (j % 8) & 1 == 1 {
                        *w |= 1 << k;
                    }
                }
            }
            let (dets, obs) = self.apply(&fired);
            write_rows(&dets, here, &mut d);
            write_rows(&obs, here, &mut o);
        }
        Ok((d, o))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_and_replay() {
        let dem = Dem::parse("error(0.25) D0 D1\nerror(0.5) D1 L0\nerror(0) D2").unwrap();
        let s = DemSampler::new(&dem);
        let shots = 20_000;
        let (d, o, e) = s.sample_seeded(3, 0, shots, 2, true);
        let rate = |rows: &[u8], stride: usize, bit: usize| (0..shots).filter(|k| rows[k * stride + bit / 8] >> (bit % 8) & 1 == 1).count() as f64 / shots as f64;
        assert!((rate(&d, 1, 0) - 0.25).abs() < 0.02);
        assert!((rate(&d, 1, 1) - 0.5).abs() < 0.02, "D1 fires when exactly one fault does: 0.25·0.5 + 0.75·0.5");
        assert_eq!(rate(&d, 1, 2), 0.0);
        assert!((rate(&o, 1, 0) - 0.5).abs() < 0.02);
        let (rd, ro) = s.replay(&e, shots).unwrap();
        assert_eq!((rd, ro), (d.clone(), o));
        assert_eq!(s.sample_seeded(3, 0, shots, 1, false).0, d, "threads do not change the shots");
    }
}

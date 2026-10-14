//! Bit flipping (Gallager's algorithm), as `ldpc`'s `FlipDecoder` (src_cpp/flip.hpp): each pass
//! flips every bit with more unsatisfied checks than satisfied ones, in column order, and
//! (every `pfreq`-th pass) flips a tied bit with probability ½. `ldpc`'s `BpFlipDecoder` runs
//! it before BP, BP decoding what it leaves.

use crate::cpp_random::Mt19937;

pub struct Flip {
    num_checks: usize,
    columns: Vec<Vec<u32>>,
    pub max_iter: usize,
    pub pfreq: usize,
    rng: Mt19937,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlipOutcome {
    pub converged: bool,
    pub iterations: usize,
}

impl Flip {
    /// `max_iter` 0 means the block length, `pfreq` 0 never, `seed` 0 the clock (as `ldpc`).
    pub fn new(num_checks: usize, columns: Vec<Vec<u32>>, max_iter: usize, pfreq: usize, seed: u32) -> Flip {
        let n = columns.len();
        let seed = if seed == 0 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u32) } else { seed };
        let columns = columns
            .into_iter()
            .map(|mut c| {
                c.sort_unstable();
                c.dedup();
                c
            })
            .collect();
        Flip { num_checks, columns, max_iter: if max_iter == 0 { n } else { max_iter }, pfreq: if pfreq == 0 { usize::MAX } else { pfreq }, rng: Mt19937::new(seed) }
    }

    /// `std::uniform_real_distribution<double>(0, 1)` over mt19937, as libc++ and libstdc++
    /// draw it: two outputs, the second the high word.
    fn random_double(&mut self) -> f64 {
        let r = 4_294_967_296.0f64;
        let s = f64::from(self.rng.next_u32()) + f64::from(self.rng.next_u32()) * r;
        let x = s / (r * r);
        if x >= 1.0 { 1.0 - f64::EPSILON / 2.0 } else { x }
    }

    /// Decodes a syndrome (one byte per check) into `decoding` (one byte per bit).
    pub fn decode(&mut self, syndrome: &[u8], decoding: &mut [u8]) -> FlipOutcome {
        assert_eq!(syndrome.len(), self.num_checks, "one syndrome bit per check");
        decoding.fill(0);
        let mut s: Vec<u8> = syndrome.iter().map(|&b| u8::from(b != 0)).collect();
        let mut weight: i64 = s.iter().map(|&b| i64::from(b)).sum();
        for iter in 1..=self.max_iter {
            for bit in 0..self.columns.len() {
                let unsatisfied = self.columns[bit].iter().filter(|&&c| s[c as usize] == 1).count();
                let satisfied = self.columns[bit].len() - unsatisfied;
                let flip = if satisfied < unsatisfied {
                    true
                } else if iter % self.pfreq == 0 && satisfied == unsatisfied {
                    self.random_double() < 0.5
                } else {
                    continue;
                };
                if flip {
                    decoding[bit] ^= 1;
                    for &c in &self.columns[bit] {
                        let c = c as usize;
                        weight += if s[c] == 1 { -1 } else { 1 };
                        s[c] ^= 1;
                    }
                }
                if weight == 0 {
                    return FlipOutcome { converged: true, iterations: iter };
                }
            }
        }
        FlipOutcome { converged: false, iterations: self.max_iter }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_flip_is_undone() {
        // A repetition code: bit 2 flipped fires checks 1 and 2, both of its checks.
        let columns: Vec<Vec<u32>> = vec![vec![0], vec![0, 1], vec![1, 2], vec![2, 3], vec![3]];
        let mut f = Flip::new(4, columns, 0, 0, 1);
        let mut d = vec![0u8; 5];
        let out = f.decode(&[0, 1, 1, 0], &mut d);
        assert!(out.converged);
        assert_eq!(d, vec![0, 0, 1, 0, 0]);
    }

    #[test]
    fn uniform_reals_are_the_standards() {
        // std::uniform_real_distribution<double>(0, 1) over std::mt19937(5489), from a program
        // built with Apple clang.
        let mut f = Flip::new(1, vec![vec![0]], 1, 0, 5489);
        for want in [0.1354770042967805, 0.8350085899945795, 0.96886777112423139] {
            assert_eq!(f.random_double(), want);
        }
    }
}

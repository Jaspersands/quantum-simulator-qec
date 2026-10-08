//! `rand` 0.8's `StdRng` and `Uniform<f64>`, without the dependency.
//!
//! WHY THIS EXISTS
//! ---------------
//! Relay-BP (`relay`) draws a fresh memory strength for every fault on every leg, uniformly
//! from an interval. IBM's implementation (`relay_bp`) draws them with `rand` 0.8's `StdRng`
//! seeded by `seed_from_u64` and `Uniform<f64>`, so to give its results for a seed, this draws
//! the same numbers:
//!
//! - `StdRng` is ChaCha12 (`rand_chacha` 0.3): the standard ChaCha block function with 12
//!   rounds, a 256-bit key, a 64-bit block counter and a 64-bit stream number (0), its
//!   keystream read as little-endian 32-bit words, a `u64` as two of them, low word first;
//! - `seed_from_u64` (`rand_core` 0.6) fills the key with PCG32 outputs of the seed;
//! - `Uniform::new(low, high)` shrinks `high - low` until `low + scale * max` stays below
//!   `high`, and each sample puts a draw's top 52 bits in the mantissa of a number in [1, 2),
//!   subtracts 1, scales and shifts.
//!
//! The engine stays free of dependencies, and its results cannot move under it when `rand`
//! changes.

/// ChaCha with `ROUNDS` rounds, as `rand_chacha` runs it.
#[derive(Clone, Debug)]
pub(crate) struct ChaCha<const ROUNDS: usize> {
    key: [u32; 8],
    counter: u64,
    buffer: [u32; 16],
    index: usize,
}

/// `rand` 0.8's `StdRng`.
pub(crate) type StdRng = ChaCha<12>;

fn quarter(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    s[a] = s[a].wrapping_add(s[b]);
    s[d] = (s[d] ^ s[a]).rotate_left(16);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_left(12);
    s[a] = s[a].wrapping_add(s[b]);
    s[d] = (s[d] ^ s[a]).rotate_left(8);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_left(7);
}

impl<const ROUNDS: usize> ChaCha<ROUNDS> {
    pub(crate) fn from_seed(seed: [u8; 32]) -> Self {
        let mut key = [0u32; 8];
        for (k, chunk) in key.iter_mut().zip(seed.chunks_exact(4)) {
            *k = u32::from_le_bytes(chunk.try_into().unwrap());
        }
        ChaCha { key, counter: 0, buffer: [0; 16], index: 16 }
    }

    /// `rand_core` 0.6's `SeedableRng::seed_from_u64`: the seed's bytes from PCG32.
    pub(crate) fn seed_from_u64(mut state: u64) -> Self {
        const MUL: u64 = 6364136223846793005;
        const INC: u64 = 11634580027462260723;
        let mut seed = [0u8; 32];
        for chunk in seed.chunks_exact_mut(4) {
            state = state.wrapping_mul(MUL).wrapping_add(INC);
            let xorshifted = (((state >> 18) ^ state) >> 27) as u32;
            let rot = (state >> 59) as u32;
            chunk.copy_from_slice(&xorshifted.rotate_right(rot).to_le_bytes());
        }
        Self::from_seed(seed)
    }

    fn block(&mut self) {
        const SIGMA: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];
        let mut init = [0u32; 16];
        init[..4].copy_from_slice(&SIGMA);
        init[4..12].copy_from_slice(&self.key);
        init[12] = self.counter as u32;
        init[13] = (self.counter >> 32) as u32;
        let mut s = init;
        for _ in 0..ROUNDS / 2 {
            quarter(&mut s, 0, 4, 8, 12);
            quarter(&mut s, 1, 5, 9, 13);
            quarter(&mut s, 2, 6, 10, 14);
            quarter(&mut s, 3, 7, 11, 15);
            quarter(&mut s, 0, 5, 10, 15);
            quarter(&mut s, 1, 6, 11, 12);
            quarter(&mut s, 2, 7, 8, 13);
            quarter(&mut s, 3, 4, 9, 14);
        }
        for (o, (x, i)) in self.buffer.iter_mut().zip(s.iter().zip(&init)) {
            *o = x.wrapping_add(*i);
        }
        self.counter = self.counter.wrapping_add(1);
        self.index = 0;
    }

    pub(crate) fn next_u32(&mut self) -> u32 {
        if self.index == 16 {
            self.block();
        }
        self.index += 1;
        self.buffer[self.index - 1]
    }

    /// Two words, low first. (`rand_core`'s block reader straddles its 64-word buffer the
    /// same way when a `u64` starts on its last word, so reading the keystream word by word
    /// gives the same values.)
    pub(crate) fn next_u64(&mut self) -> u64 {
        let lo = u64::from(self.next_u32());
        let hi = u64::from(self.next_u32());
        hi << 32 | lo
    }
}

/// `rand` 0.8's `Uniform<f64>`: samples in [low, high).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Uniform {
    low: f64,
    scale: f64,
}

impl Uniform {
    /// Panics unless `low < high` and both are finite, as `rand`'s does.
    pub(crate) fn new(low: f64, high: f64) -> Uniform {
        assert!(low < high, "Uniform::new called with `low >= high`");
        assert!(low.is_finite() && high.is_finite(), "Uniform::new called with non-finite boundaries");
        let max_rand = f64::from_bits((u64::MAX >> 12) | 1023 << 52) - 1.0;
        let mut scale = high - low;
        assert!(scale.is_finite(), "Uniform::new: range overflow");
        while scale * max_rand + low >= high {
            scale = f64::from_bits(scale.to_bits() - 1);
        }
        Uniform { low, scale }
    }

    pub(crate) fn sample<const R: usize>(&self, rng: &mut ChaCha<R>) -> f64 {
        let value1_2 = f64::from_bits((rng.next_u64() >> 12) | 1023 << 52);
        (value1_2 - 1.0) * self.scale + self.low
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ChaCha20 keystream for an all-zero key, nonce and counter (RFC 7539's first
    /// test vector; rand_chacha's `test_chacha_true_values_a`), both blocks.
    #[test]
    fn chacha20_known_answer() {
        let mut c = ChaCha::<20>::from_seed([0; 32]);
        let first: Vec<u32> = (0..16).map(|_| c.next_u32()).collect();
        assert_eq!(
            first,
            [
                0xade0b876, 0x903df1a0, 0xe56a5d40, 0x28bd8653, 0xb819d2bd, 0x1aed8da0, 0xccef36a8, 0xc70d778b, 0x7c5941da, 0x8d485751, 0x3fe02477, 0x374ad8b8,
                0xf4b8436a, 0x1ca11815, 0x69b687c3, 0x8665eeb2
            ]
        );
        let second: Vec<u32> = (0..16).map(|_| c.next_u32()).collect();
        assert_eq!(
            second,
            [
                0xbee7079f, 0x7a385155, 0x7c97ba98, 0x0d082d73, 0xa0290fcb, 0x6965e348, 0x3e53c612, 0xed7aee32, 0x7621b729, 0x434ee69c, 0xb03371d5, 0xd539d874,
                0x281fed31, 0x45fb0a51, 0x1f0ae1ac, 0x6f4d794b
            ]
        );
    }

    /// Known answers from `rand` 0.8.5 itself (`StdRng::seed_from_u64`, `next_u64`, and
    /// `Uniform::new(-0.24f64, 0.66)` sampled 70 times, past the first 64-word buffer).
    #[test]
    fn std_rng_and_uniform_are_rands() {
        let cases: [(u64, [u64; 4], [f64; 3], f64); 3] = [
            (0, [0xbb2a3fb2cd2c6f7f, 0xc6017c948e27697b, 0x069dc102cf310a16, 0x958b761dabe5f6d0], [0.41800207427733405, 0.4561141659179143, -0.21673982918998055], -0.22375077997194887),
            (1, [0xf9681a64d3301861, 0xb0f4d125cc0d694a, 0x6d8fc15a3248c9da, 0x2cf33517376425d3], [0.6368202635325624, 0.38211204963209366, 0.1451773033795833], -0.08469330971543823),
            (12345, [0x4e519426885dd156, 0xe213dd9eee42544b, 0x8bed72a9a6e51e67, 0x48397c829fa21100], [0.035339063135770454, 0.554804063960358, 0.25193272569373704], 0.11553916303950518),
        ];
        for (seed, words, first, seventieth) in cases {
            let mut r = StdRng::seed_from_u64(seed);
            assert_eq!(words.map(|_| r.next_u64()), words, "seed {seed}");
            let mut r = StdRng::seed_from_u64(seed);
            let u = Uniform::new(-0.24, 0.66);
            let xs: Vec<f64> = (0..70).map(|_| u.sample(&mut r)).collect();
            assert_eq!(&xs[..3], &first, "seed {seed}");
            assert_eq!(xs[69], seventieth, "seed {seed}");
        }
    }

    /// Samples stay in [low, high), and the scale is the largest that keeps them there.
    #[test]
    fn uniform_stays_in_range() {
        let u = Uniform::new(-0.24, 0.66);
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..100_000 {
            let x = u.sample(&mut rng);
            assert!((-0.24..0.66).contains(&x));
        }
        let max_rand = f64::from_bits((u64::MAX >> 12) | 1023 << 52) - 1.0;
        assert!(u.scale * max_rand + u.low < 0.66);
        assert!(f64::from_bits(u.scale.to_bits() + 1) * max_rand + u.low >= 0.66 || u.scale == 0.66 + 0.24);
    }
}

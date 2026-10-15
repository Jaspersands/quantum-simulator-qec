//! custom_rng.rs
//!
//! This module contains a custom random number generator that can be used to generate random double values between 0 and 1.
//! (`rand` 0.8's `StdRng`, `Uniform` and `SliceRandom::shuffle`, drawn as `rand` draws them, on
//! this crate's ChaCha12.)

use crate::chacha::{StdRng, Uniform};
use std::time::{SystemTime, UNIX_EPOCH};

/// `rand` 0.8's `UniformInt<usize>::new_inclusive(0, max)` sampled once (64-bit `usize`).
fn uniform_usize_inclusive(rng: &mut StdRng, max: usize) -> usize {
    let range = (max as u64).wrapping_add(1);
    if range == 0 {
        return rng.next_u64() as usize;
    }
    let ints_to_reject = (u64::MAX - range + 1) % range;
    let zone = u64::MAX - ints_to_reject;
    loop {
        let v = rng.next_u64();
        let m = (v as u128) * (range as u128);
        let (hi, lo) = ((m >> 64) as u64, m as u64);
        if lo <= zone {
            return hi as usize;
        }
    }
}

/// `rand` 0.8's `gen_range(0..ubound)`, as `SliceRandom::shuffle`'s `gen_index` calls it.
fn gen_index(rng: &mut StdRng, ubound: usize) -> usize {
    if ubound <= u32::MAX as usize {
        let range = ubound as u32;
        let zone = (range << range.leading_zeros()).wrapping_sub(1);
        loop {
            let v = rng.next_u32();
            let m = (v as u64) * (range as u64);
            let (hi, lo) = ((m >> 32) as u32, m as u32);
            if lo <= zone {
                return hi as usize;
            }
        }
    } else {
        let range = ubound as u64;
        let zone = (range << range.leading_zeros()).wrapping_sub(1);
        loop {
            let v = rng.next_u64();
            let m = (v as u128) * (range as u128);
            let (hi, lo) = ((m >> 64) as u64, m as u64);
            if lo <= zone {
                return hi as usize;
            }
        }
    }
}

fn clock_seed() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).expect("Time went backwards").as_secs()
}

/// Generates random double values between 0 and 1 using StdRng (originally Mersenne Twister) random number generator
pub struct CustomRNG {
    gen: StdRng,
}

impl CustomRNG {
    /// Constructs a new RandomNumberGenerator object with an optional seed
    /// If no seed is specified, the generator is seeded using the system clock.
    pub fn new(seed: Option<u64>) -> Self {
        CustomRNG { gen: StdRng::seed_from_u64(seed.unwrap_or_else(clock_seed)) }
    }

    /// Generates a new random double value between 0 and 1
    pub fn random_double(&mut self) -> f64 {
        Uniform::new(0.0, 1.0).sample(&mut self.gen)
    }

    /// Generates a new random integer value between 0 and max_int
    pub fn random_int(&mut self, max_int: usize) -> usize {
        uniform_usize_inclusive(&mut self.gen, max_int)
    }
}

/// A templated class for shuffling lists of data.
#[derive(Clone, Debug)]
pub struct RandomListShuffle<T> {
    generator: StdRng,
    _marker: std::marker::PhantomData<T>,
}

impl Default for RandomListShuffle<usize> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T> RandomListShuffle<T> {
    /// Default constructor
    pub fn new() -> Self {
        RandomListShuffle {
            generator: StdRng::seed_from_u64(crate::mwpf::shim::random_u64()),
            _marker: std::marker::PhantomData,
        }
    }

    /// Constructor that allows specifying a seed
    pub fn with_seed(seed: Option<u64>) -> Self {
        RandomListShuffle {
            generator: StdRng::seed_from_u64(seed.unwrap_or_else(clock_seed)),
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the seed for the random number generator
    pub fn seed(&mut self, seed: u64) {
        self.generator = StdRng::seed_from_u64(seed);
    }

    /// Shuffle a vector of data
    pub fn shuffle(&mut self, data: &mut Vec<T>) {
        for i in (1..data.len()).rev() {
            let j = gen_index(&mut self.generator, i + 1);
            data.swap(i, j);
        }
    }
}

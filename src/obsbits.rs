//! A set of logical observables of any size, as a bit mask in 64-bit words (observable k is
//! bit k % 64 of word k / 64). The engine's decoders keep observables as one `u64`; the model,
//! its builder, the bit-parallel sampler and the measurement converter take any number, as
//! Stim does.

use std::fmt;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ObsBits(pub Vec<u64>);

impl ObsBits {
    pub fn new() -> ObsBits {
        ObsBits(Vec::new())
    }

    pub fn flip(&mut self, k: usize) {
        if self.0.len() <= k / 64 {
            self.0.resize(k / 64 + 1, 0);
        }
        self.0[k / 64] ^= 1u64 << (k % 64);
    }

    pub fn bit(&self, k: usize) -> bool {
        self.0.get(k / 64).is_some_and(|w| (w >> (k % 64)) & 1 == 1)
    }

    pub fn xor_with(&mut self, other: &ObsBits) {
        if self.0.len() < other.0.len() {
            self.0.resize(other.0.len(), 0);
        }
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a ^= b;
        }
    }

    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&w| w == 0)
    }

    /// The first 64 observables as a mask (the decoders' form).
    pub fn low(&self) -> u64 {
        self.0.first().copied().unwrap_or(0)
    }

    /// Whether any observable from 64 on is set.
    pub fn has_wide(&self) -> bool {
        self.0.iter().skip(1).any(|&w| w != 0)
    }
}

impl fmt::Display for ObsBits {
    /// As `{:#b}` of the whole mask: `0b101`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let top = self.0.iter().rposition(|&w| w != 0);
        match top {
            None => f.write_str("0b0"),
            Some(t) => {
                let mut s = format!("{:b}", self.0[t]);
                for w in self.0[..t].iter().rev() {
                    s.push_str(&format!("{w:064b}"));
                }
                write!(f, "0b{s}")
            }
        }
    }
}

//! The numbers Relay-BP can run in, as IBM's `relay_bp` 0.2.2 instantiates its min-sum decoder:
//! `f32`, `f64`, the integers `i8` to `i64`, and 64-bit fixed point with 0 to 31 fractional bits
//! (`fixed::FixedI64<Frac>`).
//!
//! WHY THIS EXISTS
//! ---------------
//! Relay-BP is meant for hardware, where messages are small integers or fixed-point numbers,
//! and IBM's decoder runs in each. Its results in those types depend on exactly how each
//! converts and rounds, so each operation here is the one `relay_bp` calls, as its release
//! build performs it:
//!
//! - conversion from `f64` (`num_traits::FromPrimitive::from_f64`): `as` for `f32`;
//!   truncation toward zero for the integers, refused (`None`) when out of range or NaN; for
//!   fixed point, the nearest representable value with ties to even, refused when out of
//!   range;
//! - integer and fixed-point arithmetic wraps on overflow (release builds); fixed-point products
//!   keep the low 64 bits of the exact product shifted right by the fractional bits (rounding
//!   toward −∞), and quotients are the exact quotient truncated toward zero; division by zero
//!   panics, as it does there;
//! - `abs` and negation wrap (`abs(MIN) = MIN`), and a float's sign is its sign bit.

use std::fmt::Debug;

/// The operations `relay_bp`'s `MinSumBPDecoder<N>` uses.
pub trait RelayNum: Copy + PartialOrd + PartialEq + Debug + Send + Sync + 'static {
    /// `N::from_f64`.
    fn from_f64(x: f64) -> Option<Self>;
    /// `N::to_f64` (always defined for these types).
    fn to_f64(self) -> f64;
    fn zero() -> Self;
    fn one() -> Self;
    /// `N::max_value()`.
    fn max_value() -> Self;
    fn add(self, rhs: Self) -> Self;
    fn mul(self, rhs: Self) -> Self;
    fn div(self, rhs: Self) -> Self;
    fn neg(self) -> Self;
    fn abs(self) -> Self;
    fn is_negative(self) -> bool;
}

macro_rules! float {
    ($t:ty) => {
        impl RelayNum for $t {
            fn from_f64(x: f64) -> Option<Self> {
                Some(x as $t)
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn zero() -> Self {
                0.0
            }
            fn one() -> Self {
                1.0
            }
            fn max_value() -> Self {
                <$t>::MAX
            }
            fn add(self, rhs: Self) -> Self {
                self + rhs
            }
            fn mul(self, rhs: Self) -> Self {
                self * rhs
            }
            fn div(self, rhs: Self) -> Self {
                self / rhs
            }
            fn neg(self) -> Self {
                -self
            }
            fn abs(self) -> Self {
                <$t>::abs(self)
            }
            fn is_negative(self) -> bool {
                self.is_sign_negative()
            }
        }
    };
}
float!(f32);
float!(f64);

macro_rules! int {
    ($t:ty) => {
        impl RelayNum for $t {
            fn from_f64(x: f64) -> Option<Self> {
                // num-traits' float-to-int: truncate, then refuse what does not fit.
                if x.is_nan() {
                    return None;
                }
                let t = x.trunc();
                let (lo, hi) = (<$t>::MIN as f64, -(<$t>::MIN as f64));
                if t >= lo && t < hi {
                    Some(t as $t)
                } else {
                    None
                }
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn zero() -> Self {
                0
            }
            fn one() -> Self {
                1
            }
            fn max_value() -> Self {
                <$t>::MAX
            }
            fn add(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }
            fn mul(self, rhs: Self) -> Self {
                self.wrapping_mul(rhs)
            }
            fn div(self, rhs: Self) -> Self {
                // (Overflow, MIN / −1, panics in release builds too.)
                self / rhs
            }
            fn neg(self) -> Self {
                self.wrapping_neg()
            }
            fn abs(self) -> Self {
                self.wrapping_abs()
            }
            fn is_negative(self) -> bool {
                self < 0
            }
        }
    };
}
int!(i8);
int!(i16);
int!(i32);
int!(i64);

/// `fixed::FixedI64<F>`: an `i64` holding the value times 2^F.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Fixed<const F: u32>(pub i64);

impl<const F: u32> RelayNum for Fixed<F> {
    fn from_f64(x: f64) -> Option<Self> {
        if !x.is_finite() {
            return None;
        }
        // x·2^F is exact (a power of two); round it to the nearest integer, ties to even.
        let scaled = x * (F as f64).exp2();
        let r = scaled.round_ties_even();
        if r >= -(2f64.powi(63)) && r < 2f64.powi(63) {
            Some(Fixed(r as i64))
        } else {
            None
        }
    }
    fn to_f64(self) -> f64 {
        self.0 as f64 * (-(F as f64)).exp2()
    }
    fn zero() -> Self {
        Fixed(0)
    }
    fn one() -> Self {
        Fixed(1i64 << F)
    }
    fn max_value() -> Self {
        Fixed(i64::MAX)
    }
    fn add(self, rhs: Self) -> Self {
        Fixed(self.0.wrapping_add(rhs.0))
    }
    fn mul(self, rhs: Self) -> Self {
        Fixed(((self.0 as i128 * rhs.0 as i128) >> F) as i64)
    }
    fn div(self, rhs: Self) -> Self {
        assert!(rhs.0 != 0, "division by zero");
        Fixed((((self.0 as i128) << F) / rhs.0 as i128) as i64)
    }
    fn neg(self) -> Self {
        Fixed(self.0.wrapping_neg())
    }
    fn abs(self) -> Self {
        Fixed(self.0.wrapping_abs())
    }
    fn is_negative(self) -> bool {
        self.0 < 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversions_are_num_traits() {
        assert_eq!(i8::from_f64(127.9), Some(127));
        assert_eq!(i8::from_f64(128.0), None);
        assert_eq!(i8::from_f64(-128.7), Some(-128));
        assert_eq!(i8::from_f64(-129.0), None);
        assert_eq!(i64::from_f64(9.3e18), None);
        assert_eq!(i32::from_f64(f64::NAN), None);
        assert_eq!(Fixed::<2>::from_f64(0.375), Some(Fixed(2))); // 1.5 → 2 (even)
        assert_eq!(Fixed::<2>::from_f64(0.625), Some(Fixed(2))); // 2.5 → 2 (even)
        assert_eq!(Fixed::<2>::from_f64(-0.375), Some(Fixed(-2)));
        assert_eq!(Fixed::<0>::from_f64(f64::INFINITY), None);
    }

    #[test]
    fn fixed_arithmetic_is_fixeds() {
        // 1.25 · 1.25 = 1.5625 → 1.5 at two fractional bits (rounded down); −1.25 · 1.25 → −1.75.
        assert_eq!(Fixed::<2>(5).mul(Fixed(5)), Fixed(6));
        assert_eq!(Fixed::<2>(-5).mul(Fixed(5)), Fixed(-7));
        // 1.25 / 0.5 = 2.5; −1.25 / 1.5 = −0.833… → −0.75 (toward zero).
        assert_eq!(Fixed::<2>(5).div(Fixed(2)), Fixed(10));
        assert_eq!(Fixed::<2>(-5).div(Fixed(6)), Fixed(-3));
        assert_eq!(Fixed::<3>(i64::MIN).abs(), Fixed(i64::MIN));
        assert_eq!(RelayNum::abs(i8::MIN), i8::MIN);
        assert_eq!(100i8.add(100), -56);
        assert_eq!(Fixed::<4>(24).to_f64(), 1.5);
    }
}

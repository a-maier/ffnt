#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::{
    fmt::{self, Display},
    ops::{
        Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
    },
};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Z64<const P: u64>(u64);

impl<const P: u64> Z64<P> {
    const INFO: Z64Info = Z64Info::new(P);

    pub const MIN: Z64<P> = Self(0);
    pub const MAX: Z64<P> = Self(P - 1);

    pub const fn new(z: i64) -> Self {
        let res = remi(z, P, Self::info().red_struct);
        debug_assert!(res >= 0);
        let res = res as u64;
        debug_assert!(res < P);
        Self(res)
    }

    pub const unsafe fn new_unchecked(z: u64) -> Self {
        debug_assert!(z <= P);
        Self(z)
    }

    pub fn inv(&self) -> Self {
        self.try_inv().expect("Number has no multiplicative inverse")
    }

    pub fn try_inv(&self) -> Option<Self> {
        let res = extended_gcd(self.0, Self::modulus());
        if res.gcd != 1 {
            return None;
        }
        let s = res.bezout[0];
        Some(Self(
            if s < 0 {
                debug_assert!(s + Self::modulus() as i64 >= 0);
                s + Self::modulus() as i64
            } else {
                s
            } as u64
        ))
    }

    pub const fn has_inv(&self) -> bool {
        gcd(self.0, Self::modulus()) == 1
    }

    const fn info() -> &'static Z64Info {
        &Self::INFO
    }

    pub const fn modulus() -> u64 {
        P
    }

    pub const fn modulus_inv() -> SpInverse64 {
        Self::info().p_inv
    }

    pub fn powi(self, exp: i64) -> Self {
        if exp < 0 {
            self.powu((-exp) as u64).inv()
        } else {
            self.powu(exp as u64)
        }
    }

    pub fn powu(mut self, mut exp: u64) -> Self {
        let mut res = unsafe { Self::new_unchecked(1) };
        while exp > 0 {
            if exp & 1 != 0 {
                res *= self
            };
            self *= self;
            exp /= 2;
        }
        res
    }

    pub(crate) const fn repr(self) -> u64 {
        self.0
    }
}

impl<const P: u64> From<Z64<P>> for u64 {
    fn from(i: Z64<P>) -> Self {
        i.0
    }
}

impl<const P: u64> From<Z64<P>> for i64 {
    fn from(i: Z64<P>) -> Self {
        i.0 as i64
    }
}

impl<const P: u64> From<u64> for Z64<P> {
    fn from(u: u64) -> Self {
        Self(remu(u, Self::modulus(), Self::info().red_struct) as u64)
    }
}

impl<const P: u64> From<i64> for Z64<P> {
    fn from(i: i64) -> Self {
        Self::new(i)
    }
}

impl<const P: u64> From<i32> for Z64<P> {
    fn from(i: i32) -> Self {
        Self::from(i as i64)
    }
}

impl<const P: u64> From<u32> for Z64<P> {
    fn from(u: u32) -> Self {
        Self::from(u as u64)
    }
}

impl<const P: u64> From<i16> for Z64<P> {
    fn from(i: i16) -> Self {
        Self::from(i as i64)
    }
}

impl<const P: u64> From<u16> for Z64<P> {
    fn from(u: u16) -> Self {
        Self::from(u as u64)
    }
}

impl<const P: u64> From<i8> for Z64<P> {
    fn from(i: i8) -> Self {
        Self::from(i as i64)
    }
}

impl<const P: u64> From<u8> for Z64<P> {
    fn from(u: u8) -> Self {
        Self::from(u as u64)
    }
}

impl<const P: u64> Display for Z64<P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<const P: u64> AddAssign for Z64<P> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u64> AddAssign<&Z64<P>> for Z64<P> {
    fn add_assign(&mut self, rhs: &Self) {
        *self = *self + *rhs;
    }
}

impl<const P: u64> SubAssign for Z64<P> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u64> SubAssign<&Z64<P>> for Z64<P> {
    fn sub_assign(&mut self, rhs: &Self) {
        *self -= *rhs;
    }
}

impl<const P: u64> MulAssign for Z64<P> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u64> MulAssign<&Z64<P>> for Z64<P> {
    fn mul_assign(&mut self, rhs: &Self) {
        *self = *self * *rhs;
    }
}

impl<const P: u64> DivAssign for Z64<P> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<const P: u64> DivAssign<&Z64<P>> for Z64<P> {
    fn div_assign(&mut self, rhs: &Self) {
        *self = *self / *rhs;
    }
}

impl<const P: u64> Add for Z64<P> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let res = correct_excess((self.0 + rhs.0) as i64, Self::modulus());
        debug_assert!(res >= 0);
        let res = res as u64;
        debug_assert!(res < Self::modulus());
        Self(res)
    }
}

impl<const P: u64> Add for &Z64<P> {
    type Output = Z64<P>;

    fn add(self, rhs: Self) -> Self::Output {
        *self + *rhs
    }
}

impl<const P: u64> Add<Z64<P>> for &Z64<P> {
    type Output = Z64<P>;

    fn add(self, rhs: Z64<P>) -> Self::Output {
        *self + rhs
    }
}

impl<const P: u64> Add<&Z64<P>> for Z64<P> {
    type Output = Z64<P>;

    fn add(self, rhs: &Z64<P>) -> Self::Output {
        self + *rhs
    }
}

impl<const P: u64> Sub for Z64<P> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let res =
            correct_deficit(self.0 as i64 - rhs.0 as i64, Self::modulus());
        debug_assert!(res >= 0);
        let res = res as u64;
        debug_assert!(res < Self::modulus());
        Self(res)
    }
}

impl<const P: u64> Sub for &Z64<P> {
    type Output = Z64<P>;

    fn sub(self, rhs: Self) -> Self::Output {
        *self - *rhs
    }
}

impl<const P: u64> Sub<Z64<P>> for &Z64<P> {
    type Output = Z64<P>;

    fn sub(self, rhs: Z64<P>) -> Self::Output {
        *self - rhs
    }
}

impl<const P: u64> Sub<&Z64<P>> for Z64<P> {
    type Output = Z64<P>;

    fn sub(self, rhs: &Z64<P>) -> Self::Output {
        self - *rhs
    }
}

impl<const P: u64> Neg for Z64<P> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::default() - self
    }
}

impl<const P: u64> Mul for Z64<P> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(mul_mod(self.0, rhs.0, Self::modulus(), Self::modulus_inv()))
    }
}

impl<const P: u64> Mul for &Z64<P> {
    type Output = Z64<P>;

    fn mul(self, rhs: Self) -> Self::Output {
        *self * *rhs
    }
}

impl<const P: u64> Mul<Z64<P>> for &Z64<P> {
    type Output = Z64<P>;

    fn mul(self, rhs: Z64<P>) -> Self::Output {
        *self * rhs
    }
}

impl<const P: u64> Mul<&Z64<P>> for Z64<P> {
    type Output = Z64<P>;

    fn mul(self, rhs: &Z64<P>) -> Self::Output {
        self * *rhs
    }
}

impl<const P: u64> Div for Z64<P> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inv()
    }
}

const fn mul_mod(a: u64, b: u64, n: u64, ninv: SpInverse64) -> u64 {
    let res = normalised_mul_mod(
        a,
        (b as i64) << ninv.shamt,
        ((n as i64) << ninv.shamt) as u64,
        ninv.inv,
    ) >> ninv.shamt;
    res as u64
}

impl<const P: u64> Div for &Z64<P> {
    type Output = Z64<P>;

    fn div(self, rhs: Self) -> Self::Output {
        *self / *rhs
    }
}

impl<const P: u64> Div<Z64<P>> for &Z64<P> {
    type Output = Z64<P>;

    fn div(self, rhs: Z64<P>) -> Self::Output {
        *self / rhs
    }
}

impl<const P: u64> Div<&Z64<P>> for Z64<P> {
    type Output = Z64<P>;

    fn div(self, rhs: &Z64<P>) -> Self::Output {
        self / *rhs
    }
}

/// Fallible division
pub trait TryDiv<Rhs = Self> {
    /// Result type of successfull division
    type Output;

    /// Tries to divide by the argument.
    ///
    /// `a.try_div(b)` returns `Some(a / b)` if the division
    /// is successful and `None` otherwise.
    #[must_use]
    fn try_div(self, rhs: Rhs) -> Option<Self::Output>;
}

impl<const P: u64> TryDiv for Z64<P> {
    type Output = Self;

    fn try_div(self, rhs: Self) -> Option<Self::Output> {
        rhs.try_inv().map(|i| self * i)
    }
}

impl<const P: u64> TryDiv for &Z64<P> {
    type Output = Z64<P>;

    fn try_div(self, rhs: Self) -> Option<Self::Output> {
        (*self).try_div(*rhs)
    }
}

impl<const P: u64> TryDiv<Z64<P>> for &Z64<P> {
    type Output = Z64<P>;

    fn try_div(self, rhs: Z64<P>) -> Option<Self::Output> {
        (*self).try_div(rhs)
    }
}

impl<const P: u64> TryDiv<&Z64<P>> for Z64<P> {
    type Output = Z64<P>;

    fn try_div(self, rhs: &Z64<P>) -> Option<Self::Output> {
        self.try_div(*rhs)
    }
}

const fn normalised_mul_mod(a: u64, b: i64, n: u64, ninv: u64) -> i64 {
    let u = a as u128 * b as u128;
    let h = (u >> (SP_NBITS - 2)) as u64;
    let q = u128_mul_high(h, ninv);
    // no shift needed, NTL_POST_SHIFT is 0
    let l = u as u64;
    let r = l.wrapping_sub(q.wrapping_mul(n));
    debug_assert!(r < 2 * n);
    correct_excess(r as i64, n)
}

const fn remu(z: u64, p: u64, red: ReduceStruct) -> i64 {
    let q = u128_mul_high(z, red.ninv);
    let qp = q.wrapping_mul(p);
    let r = z as i64 - qp as i64;
    correct_excess(r, p)
}

const fn remi(z: i64, p: u64, red: ReduceStruct) -> i64 {
    let zu = (z as u64) & ((1u64 << (u64::BITS - 1)) - 1);
    let r = remu(zu, p, red);
    let s = i64_sign_mask(z) & (red.sgn as i64);
    correct_deficit(r - s, p)
}

const fn u128_mul_high(a: u64, b: u64) -> u64 {
    u128_get_high(a as u128 * b as u128)
}

const fn u128_get_high(u: u128) -> u64 {
    (u >> u64::BITS) as u64
}

const fn correct_excess(a: i64, p: u64) -> i64 {
    let n = p as i64;
    (a - n) + (i64_sign_mask(a - n) & n)
}

const fn correct_deficit(a: i64, p: u64) -> i64 {
    a + (i64_sign_mask(a) & (p as i64))
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct ExtendedGCDResult {
    gcd: u64,
    bezout: [i64; 2],
}

const fn extended_gcd(a: u64, b: u64) -> ExtendedGCDResult {
    let mut old_r = a;
    let mut r = b;
    let mut old_s = 1;
    let mut s = 0;
    let mut old_t = 0;
    let mut t = 1;

    while r != 0 {
        let quotient = old_r / r;
        (old_r, r) = (r, old_r - quotient * r);
        (old_s, s) = (s, old_s - quotient as i64 * s);
        (old_t, t) = (t, old_t - quotient as i64 * t);
    }
    ExtendedGCDResult {
        gcd: old_r,
        bezout: [old_s, old_t],
    }
}

const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b)
    }
    a
}

const SP_NBITS: u32 = u64::BITS - 4;
const PRE_SHIFT2: u32 = 2 * SP_NBITS + 2;

const fn used_bits(z: u64) -> u32 {
    u64::BITS - z.leading_zeros()
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct Z64Info {
    p: u64,
    p_inv: SpInverse64,
    red_struct: ReduceStruct,
}

impl Z64Info {
    const fn new(p: u64) -> Self {
        assert!(p > 1);
        assert!(used_bits(p) <= SP_NBITS);

        let p_inv = prep_mul_mod(p);
        let red_struct = prep_rem(p);
        Self {
            p,
            p_inv,
            red_struct,
        }
    }
}

const fn prep_mul_mod(p: u64) -> SpInverse64 {
    let shamt = p.leading_zeros() - (u64::BITS - SP_NBITS);
    let inv = normalised_prep_mul_mod(p << shamt);
    SpInverse64 { inv, shamt }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct ReduceStruct {
    ninv: u64,
    sgn: u64,
}

const fn prep_rem(p: u64) -> ReduceStruct {
    let mut q = (1 << (u64::BITS - 1)) / p;
    // r = 2^63 % p
    let r = (1 << (u64::BITS - 1)) - q * p;

    q *= 2;
    q += correct_excess_quo(2 * r as i64, p as i64).0;

    ReduceStruct { ninv: q, sgn: r }
}

const fn correct_excess_quo(a: i64, n: i64) -> (u64, i64) {
    if a >= n {
        (1, a - n)
    } else {
        (0, a)
    }
}

// TODO: make const as soon as allowed by rust
trait SignMask {
    fn sign_mask(self) -> i64;
}

impl SignMask for u64 {
    fn sign_mask(self) -> i64 {
        u64_sign_mask(self)
    }
}

impl SignMask for i64 {
    fn sign_mask(self) -> i64 {
        i64_sign_mask(self)
    }
}

const fn i64_sign_mask(i: i64) -> i64 {
    i >> (u64::BITS - 1)
}

const fn u64_sign_mask(i: u64) -> i64 {
    i64_sign_mask(i as i64)
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SpInverse64 {
    inv: u64,
    shamt: u32,
}

// Adapted from NTL's sp_NormalizedPrepMulMod
//
// Floating-point arithmetic replaced be u128 / i128 to allow `const`.
// The performance impact is not a huge concern since this function
// is only evaluated at compile time and only once for each prime field order.
// This is unlike NTL, where each change triggers a recalculation?
//
// This only works since this function is `const` and can be therefore
// used to compute individual `const INFO` inside `Z64<P>` for each
// `P`. The alternatives `lazy_static!` or `OnceCell` would not be
// recomputed, but instead incorrectly shared between `Z64<P>` with
// different `P`!
const fn normalised_prep_mul_mod(n: u64) -> u64 {
    // NOTE: this is an initial approximation
    //       the true quotient is <= 2^SP_NBITS
    const MAX: u128 = 1u128 << (2 * SP_NBITS - 1);
    let init_quot_approx = MAX / n as u128;

    let approx_rem = MAX - n as u128 * init_quot_approx;

    let approx_rem = (approx_rem << (PRE_SHIFT2 - 2 * SP_NBITS + 1)) - 1;

    let approx_rem_low = approx_rem as u64;
    let s1 = (approx_rem >> u64::BITS) as u64;
    let s2 = approx_rem_low >> (u64::BITS - 1);
    let approx_rem_high = s1.wrapping_add(s2);

    let approx_rem_low = approx_rem_low as i64;
    let approx_rem_high = approx_rem_high as i64;

    let bpl = 1i128 << u64::BITS;

    let fr = approx_rem_low as i128 + approx_rem_high as i128 * bpl;

    // now convert fr*ninv to a long
    // but we have to be careful: fr may be negative.
    // the result should still give floor(r/n) pm 1,
    // and is computed in a way that avoids branching

    let mut q1 = (fr / n as i128) as i64;
    if q1 < 0 {
        // This counteracts the round-to-zero behavior of conversion
        // to i64.  It should be compiled into branch-free code.
        q1 -= 1
    }

    let mut q1 = q1 as u64;
    let approx_rem_low = approx_rem_low as u64;
    let sub = q1.wrapping_mul(n);

    let approx_rem = approx_rem_low.wrapping_sub(sub);

    q1 += (1
        + u64_sign_mask(approx_rem)
        + u64_sign_mask(approx_rem.wrapping_sub(n))) as u64;

    ((init_quot_approx as u64) << (PRE_SHIFT2 - 2 * SP_NBITS + 1))
        .wrapping_add(q1)

    // NTL_PRE_SHIFT1 is 0, so no further shift required
}

#[cfg(test)]
mod tests {

    use ::rand::{Rng, SeedableRng};
    use once_cell::sync::Lazy;
    use rug::{ops::Pow, Integer};

    use super::*;

    const PRIMES: [u64; 3] = [3, 443619635352171979, 1152921504606846883];

    #[test]
    fn z64_info() {
        let tst = Z64::<{ PRIMES[0] }>::info();
        assert_eq!(Z64::<{ PRIMES[0] }>::modulus(), PRIMES[0]);
        assert_eq!(tst.p_inv.inv, 6148914691236517205);
        assert_eq!(tst.p_inv.shamt, 58);
        assert_eq!(tst.red_struct.ninv, 6148914691236517205);
        assert_eq!(tst.red_struct.sgn, 2);

        let tst = Z64::<{ PRIMES[1] }>::info();
        assert_eq!(Z64::<{ PRIMES[1] }>::modulus(), PRIMES[1]);
        assert_eq!(tst.p_inv.inv, 5992647258409536587);
        assert_eq!(tst.p_inv.shamt, 1);
        assert_eq!(tst.red_struct.ninv, 41);
        assert_eq!(tst.red_struct.sgn, 350979329811336228);

        let tst = Z64::<{ PRIMES[2] }>::info();
        assert_eq!(Z64::<{ PRIMES[2] }>::modulus(), PRIMES[2]);
        assert_eq!(tst.p_inv.inv, 4611686018427388276);
        assert_eq!(tst.p_inv.shamt, 0);
        assert_eq!(tst.red_struct.ninv, 16);
        assert_eq!(tst.red_struct.sgn, 744);
    }

    #[test]
    fn z64_has_inv() {
        type Z = Z64<6>;
        assert!(!Z::from(0).has_inv());
        assert!(Z::from(1).has_inv());
        assert!(!Z::from(2).has_inv());
        assert!(!Z::from(3).has_inv());
        assert!(!Z::from(4).has_inv());
        assert!(Z::from(5).has_inv());
        assert_eq!(Z::from(6), Z::from(0));
    }

    #[test]
    #[should_panic]
    fn z64_inv0() {
        type Z = Z64<6>;
        Z::from(0).inv();
    }

    #[test]
    #[should_panic]
    fn z64_inv2() {
        type Z = Z64<6>;
        Z::from(2).inv();
    }

    #[test]
    fn z64_constr() {
        let z: Z64<3> = 2.into();
        assert_eq!(u64::from(z), 2);
        let z: Z64<3> = (-1).into();
        assert_eq!(u64::from(z), 2);
        let z: Z64<3> = 5.into();
        assert_eq!(u64::from(z), 2);

        let z: Z64<3> = 0.into();
        assert_eq!(u64::from(z), 0);
        let z: Z64<3> = 3.into();
        assert_eq!(u64::from(z), 0);

        let z: Z64<3> = 2u32.into();
        assert_eq!(u64::from(z), 2);
        let z: Z64<3> = 5u32.into();
        assert_eq!(u64::from(z), 2);

        let z: Z64<3> = 0u32.into();
        assert_eq!(u64::from(z), 0);
        let z: Z64<3> = 3u32.into();
        assert_eq!(u64::from(z), 0);
    }

    static POINTS: Lazy<[i64; 1000]> = Lazy::new(|| {
        let mut pts = [0; 1000];
        let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(0);
        for pt in &mut pts {
            *pt = rng.gen();
        }
        pts
    });

    #[test]
    fn tst_conv() {
        for pt in *POINTS {
            let z: Z64<{ PRIMES[0] }> = pt.into();
            let z: i64 = z.into();
            assert_eq!(z, pt.rem_euclid(PRIMES[0] as i64));
        }

        for pt in *POINTS {
            let z: Z64<{ PRIMES[1] }> = pt.into();
            let z: i64 = z.into();
            assert_eq!(z, pt.rem_euclid(PRIMES[1] as i64));
        }

        for pt in *POINTS {
            let z: Z64<{ PRIMES[2] }> = pt.into();
            let z: i64 = z.into();
            assert_eq!(z, pt.rem_euclid(PRIMES[2] as i64));
        }
    }

    #[test]
    fn tst_add() {
        for pt1 in *POINTS {
            let z1: Z64<{ PRIMES[0] }> = pt1.into();
            let pt1 = pt1 as i128;
            for pt2 in *POINTS {
                let z2: Z64<{ PRIMES[0] }> = pt2.into();
                let pt2 = pt2 as i128;
                let sum1: i64 = (z1 + z2).into();
                let sum2 = (pt1 + pt2).rem_euclid(PRIMES[0] as i128) as i64;
                assert_eq!(sum1, sum2);
            }
        }

        for pt1 in *POINTS {
            let z1: Z64<{ PRIMES[1] }> = pt1.into();
            let pt1 = pt1 as i128;
            for pt2 in *POINTS {
                let z2: Z64<{ PRIMES[1] }> = pt2.into();
                let pt2 = pt2 as i128;
                let sum1: i64 = (z1 + z2).into();
                let sum2 = (pt1 + pt2).rem_euclid(PRIMES[1] as i128) as i64;
                assert_eq!(sum1, sum2);
            }
        }

        for pt1 in *POINTS {
            let z1: Z64<{ PRIMES[2] }> = pt1.into();
            let pt1 = pt1 as i128;
            for pt2 in *POINTS {
                let z2: Z64<{ PRIMES[2] }> = pt2.into();
                let pt2 = pt2 as i128;
                let sum1: i64 = (z1 + z2).into();
                let sum2 = (pt1 + pt2).rem_euclid(PRIMES[2] as i128) as i64;
                assert_eq!(sum1, sum2);
            }
        }
    }

    #[test]
    fn tst_sub() {
        for pt1 in *POINTS {
            let z1: Z64<{ PRIMES[0] }> = pt1.into();
            let pt1 = pt1 as i128;
            for pt2 in *POINTS {
                let z2: Z64<{ PRIMES[0] }> = pt2.into();
                let pt2 = pt2 as i128;
                let sum1: i64 = (z1 - z2).into();
                let sum2 = (pt1 - pt2).rem_euclid(PRIMES[0] as i128) as i64;
                assert_eq!(sum1, sum2);
            }
        }

        for pt1 in *POINTS {
            let z1: Z64<{ PRIMES[1] }> = pt1.into();
            let pt1 = pt1 as i128;
            for pt2 in *POINTS {
                let z2: Z64<{ PRIMES[1] }> = pt2.into();
                let pt2 = pt2 as i128;
                let sum1: i64 = (z1 - z2).into();
                let sum2 = (pt1 - pt2).rem_euclid(PRIMES[1] as i128) as i64;
                assert_eq!(sum1, sum2);
            }
        }

        for pt1 in *POINTS {
            let z1: Z64<{ PRIMES[2] }> = pt1.into();
            let pt1 = pt1 as i128;
            for pt2 in *POINTS {
                let z2: Z64<{ PRIMES[2] }> = pt2.into();
                let pt2 = pt2 as i128;
                let sum1: i64 = (z1 - z2).into();
                let sum2 = (pt1 - pt2).rem_euclid(PRIMES[2] as i128) as i64;
                assert_eq!(sum1, sum2);
            }
        }
    }

    #[test]
    fn tst_mul() {
        for pt1 in *POINTS {
            let z1: Z64<{ PRIMES[0] }> = pt1.into();
            let pt1 = pt1 as i128;
            for pt2 in *POINTS {
                let z2: Z64<{ PRIMES[0] }> = pt2.into();
                let pt2 = pt2 as i128;
                let prod1: i64 = (z1 * z2).into();
                let prod2 = (pt1 * pt2).rem_euclid(PRIMES[0] as i128) as i64;
                assert_eq!(prod1, prod2);
            }
        }

        for pt1 in *POINTS {
            let z1: Z64<{ PRIMES[1] }> = pt1.into();
            let pt1 = pt1 as i128;
            for pt2 in *POINTS {
                let z2: Z64<{ PRIMES[1] }> = pt2.into();
                let pt2 = pt2 as i128;
                let prod1: i64 = (z1 * z2).into();
                let prod2 = (pt1 * pt2).rem_euclid(PRIMES[1] as i128) as i64;
                assert_eq!(prod1, prod2);
            }
        }

        for pt1 in *POINTS {
            let z1: Z64<{ PRIMES[2] }> = pt1.into();
            let pt1 = pt1 as i128;
            for pt2 in *POINTS {
                let z2: Z64<{ PRIMES[2] }> = pt2.into();
                let pt2 = pt2 as i128;
                let prod1: i64 = (z1 * z2).into();
                let prod2 = (pt1 * pt2).rem_euclid(PRIMES[2] as i128) as i64;
                assert_eq!(prod1, prod2);
            }
        }
    }

    #[test]
    fn tst_div() {
        for pt1 in *POINTS {
            let z1: Z64<{ PRIMES[0] }> = pt1.into();
            for pt2 in *POINTS {
                let z2: Z64<{ PRIMES[0] }> = pt2.into();
                if i64::from(z2) == 0 {
                    continue;
                }
                let div = z1 / z2;
                assert_eq!(z1, div * z2);
            }
        }

        for pt1 in *POINTS {
            let z1: Z64<{ PRIMES[1] }> = pt1.into();
            for pt2 in *POINTS {
                let z2: Z64<{ PRIMES[1] }> = pt2.into();
                if i64::from(z2) == 0 {
                    continue;
                }
                let div = z1 / z2;
                assert_eq!(z1, div * z2);
            }
        }

        for pt1 in *POINTS {
            let z1: Z64<{ PRIMES[2] }> = pt1.into();
            for pt2 in *POINTS {
                let z2: Z64<{ PRIMES[2] }> = pt2.into();
                if i64::from(z2) == 0 {
                    continue;
                }
                let div = z1 / z2;
                assert_eq!(z1, div * z2);
            }
        }
    }

    #[test]
    fn tst_pow() {
        let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(2849);
        for pt1 in *POINTS {
            let base = Integer::from(pt1);
            for _ in 0..100 {
                let exp: u8 = rng.gen();
                let pow = base.clone().pow(exp as u32);
                // ensure remainder is positive and less than the mod
                let ref_pow0 =
                    (pow.clone() % PRIMES[0] + PRIMES[0]) % PRIMES[0];
                let ref_pow0: u64 = ref_pow0.try_into().unwrap();
                let z: Z64<{ PRIMES[0] }> = pt1.into();
                let pow0: u64 = z.powu(exp as u64).into();
                assert_eq!(pow0, ref_pow0);

                let ref_pow0 =
                    (pow.clone() % PRIMES[1] + PRIMES[1]) % PRIMES[1];
                let ref_pow0: u64 = ref_pow0.try_into().unwrap();
                let z: Z64<{ PRIMES[1] }> = pt1.into();
                let pow0: u64 = z.powu(exp as u64).into();
                assert_eq!(pow0, ref_pow0);

                let ref_pow0 = (pow % PRIMES[2] + PRIMES[2]) % PRIMES[2];
                let ref_pow0: u64 = ref_pow0.try_into().unwrap();
                let z: Z64<{ PRIMES[2] }> = pt1.into();
                let pow0: u64 = z.powu(exp as u64).into();
                assert_eq!(pow0, ref_pow0);
            }
        }
    }
}

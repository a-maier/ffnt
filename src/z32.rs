// TODO: code duplication with z64
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
pub struct Z32<const P: u32>(pub(crate) u32);

impl<const P: u32> Z32<P> {
    const INFO: Z32Info = Z32Info::new(P);

    pub const fn new(z: i32) -> Self {
        let res = remi(z, P, Self::info().red_struct);
        debug_assert!(res >= 0);
        let res = res as u32;
        debug_assert!(res < P);
        Self(res)
    }

    pub const fn new_unchecked(z: u32) -> Self {
        debug_assert!(z <= P);
        Self(z)
    }

    pub fn inv(&self) -> Self {
        let res = extended_gcd(self.0, Self::modulus());
        assert_eq!(res.gcd, 1, "inverse undefined for {}", self.0);
        let s = res.bezout[0];
        if s < 0 {
            debug_assert!(s + Self::modulus() as i32 >= 0);
            Self((s + Self::modulus() as i32) as u32)
        } else {
            Self(s as u32)
        }
    }

    pub const fn has_inv(&self) -> bool {
        gcd(self.0, Self::modulus()) == 1
    }

    const fn info() -> &'static Z32Info {
        &Self::INFO
    }

    pub const fn modulus() -> u32 {
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
        let mut res = Self::new_unchecked(1);
        while exp > 0 {
            if exp & 1 != 0 {
                res *= self
            };
            self *= self;
            exp /= 2;
        }
        res
    }
}

impl<const P: u32> From<Z32<P>> for u32 {
    fn from(i: Z32<P>) -> Self {
        i.0
    }
}

impl<const P: u32> From<Z32<P>> for i32 {
    fn from(i: Z32<P>) -> Self {
        i.0 as i32
    }
}

impl<const P: u32> From<u32> for Z32<P> {
    fn from(u: u32) -> Self {
        Self(remu(u, Self::modulus(), Self::info().red_struct) as u32)
    }
}

impl<const P: u32> From<i32> for Z32<P> {
    fn from(i: i32) -> Self {
        Self::new(i as i32)
    }
}

impl<const P: u32> From<i16> for Z32<P> {
    fn from(i: i16) -> Self {
        Self::from(i as i32)
    }
}

impl<const P: u32> From<u16> for Z32<P> {
    fn from(u: u16) -> Self {
        Self::from(u as u32)
    }
}

impl<const P: u32> From<i8> for Z32<P> {
    fn from(i: i8) -> Self {
        Self::from(i as i32)
    }
}

impl<const P: u32> From<u8> for Z32<P> {
    fn from(u: u8) -> Self {
        Self::from(u as u32)
    }
}

impl<const P: u32> Display for Z32<P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<const P: u32> AddAssign for Z32<P> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u32> AddAssign<&Z32<P>> for Z32<P> {
    fn add_assign(&mut self, rhs: &Self) {
        *self = *self + *rhs;
    }
}

impl<const P: u32> SubAssign for Z32<P> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u32> SubAssign<&Z32<P>> for Z32<P> {
    fn sub_assign(&mut self, rhs: &Self) {
        *self -= *rhs;
    }
}

impl<const P: u32> MulAssign for Z32<P> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u32> MulAssign<&Z32<P>> for Z32<P> {
    fn mul_assign(&mut self, rhs: &Self) {
        *self = *self * *rhs;
    }
}

impl<const P: u32> DivAssign for Z32<P> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<const P: u32> DivAssign<&Z32<P>> for Z32<P> {
    fn div_assign(&mut self, rhs: &Self) {
        *self = *self / *rhs;
    }
}

impl<const P: u32> Add for Z32<P> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let res = correct_excess((self.0 + rhs.0) as i32, Self::modulus());
        debug_assert!(res >= 0);
        let res = res as u32;
        debug_assert!(res < Self::modulus());
        Self(res)
    }
}

impl<const P: u32> Add for &Z32<P> {
    type Output = Z32<P>;

    fn add(self, rhs: Self) -> Self::Output {
        *self + *rhs
    }
}

impl<const P: u32> Add<Z32<P>> for &Z32<P> {
    type Output = Z32<P>;

    fn add(self, rhs: Z32<P>) -> Self::Output {
        *self + rhs
    }
}

impl<const P: u32> Add<&Z32<P>> for Z32<P> {
    type Output = Z32<P>;

    fn add(self, rhs: &Z32<P>) -> Self::Output {
        self + *rhs
    }
}

impl<const P: u32> Sub for Z32<P> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let res =
            correct_deficit(self.0 as i32 - rhs.0 as i32, Self::modulus());
        debug_assert!(res >= 0);
        let res = res as u32;
        debug_assert!(res < Self::modulus());
        Self(res)
    }
}

impl<const P: u32> Sub for &Z32<P> {
    type Output = Z32<P>;

    fn sub(self, rhs: Self) -> Self::Output {
        *self - *rhs
    }
}

impl<const P: u32> Sub<Z32<P>> for &Z32<P> {
    type Output = Z32<P>;

    fn sub(self, rhs: Z32<P>) -> Self::Output {
        *self - rhs
    }
}

impl<const P: u32> Sub<&Z32<P>> for Z32<P> {
    type Output = Z32<P>;

    fn sub(self, rhs: &Z32<P>) -> Self::Output {
        self - *rhs
    }
}

impl<const P: u32> Neg for Z32<P> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::default() - self
    }
}

impl<const P: u32> Mul for Z32<P> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(mul_mod(self.0, rhs.0, Self::modulus(), Self::modulus_inv()))
    }
}

impl<const P: u32> Mul for &Z32<P> {
    type Output = Z32<P>;

    fn mul(self, rhs: Self) -> Self::Output {
        *self * *rhs
    }
}

impl<const P: u32> Mul<Z32<P>> for &Z32<P> {
    type Output = Z32<P>;

    fn mul(self, rhs: Z32<P>) -> Self::Output {
        *self * rhs
    }
}

impl<const P: u32> Mul<&Z32<P>> for Z32<P> {
    type Output = Z32<P>;

    fn mul(self, rhs: &Z32<P>) -> Self::Output {
        self * *rhs
    }
}

impl<const P: u32> Div for Z32<P> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inv()
    }
}

const fn mul_mod(a: u32, b: u32, n: u32, ninv: SpInverse64) -> u32 {
    let res = normalised_mul_mod(
        a,
        (b as i32) << ninv.shamt,
        ((n as i32) << ninv.shamt) as u32,
        ninv.inv,
    ) >> ninv.shamt;
    res as u32
}

impl<const P: u32> Div for &Z32<P> {
    type Output = Z32<P>;

    fn div(self, rhs: Self) -> Self::Output {
        *self / *rhs
    }
}

impl<const P: u32> Div<Z32<P>> for &Z32<P> {
    type Output = Z32<P>;

    fn div(self, rhs: Z32<P>) -> Self::Output {
        *self / rhs
    }
}

impl<const P: u32> Div<&Z32<P>> for Z32<P> {
    type Output = Z32<P>;

    fn div(self, rhs: &Z32<P>) -> Self::Output {
        self / *rhs
    }
}

const fn normalised_mul_mod(a: u32, b: i32, n: u32, ninv: u32) -> i32 {
    let u = a as u64 * b as u64;
    let h = (u >> (SP_NBITS - 2)) as u32;
    let q = u64_mul_high(h, ninv) >> POST_SHIFT;
    let l = u as u32;
    let r = l.wrapping_sub(q.wrapping_mul(n));
    debug_assert!(r < 2 * n);
    correct_excess(r as i32, n)
}

const fn remu(z: u32, p: u32, red: ReduceStruct) -> i32 {
    let q = u64_mul_high(z, red.ninv);
    let qp = q.wrapping_mul(p);
    let r = z as i32 - qp as i32;
    correct_excess(r, p)
}

const fn remi(z: i32, p: u32, red: ReduceStruct) -> i32 {
    let zu = (z as u32) & ((1u32 << (u32::BITS - 1)) - 1);
    let r = remu(zu, p, red);
    let s = i32_sign_mask(z) & (red.sgn as i32);
    correct_deficit(r - s, p)
}

const fn u64_mul_high(a: u32, b: u32) -> u32 {
    u64_get_high(a as u64 * b as u64)
}

const fn u64_get_high(u: u64) -> u32 {
    (u >> u32::BITS) as u32
}

const fn correct_excess(a: i32, p: u32) -> i32 {
    let n = p as i32;
    (a - n) + (i32_sign_mask(a - n) & n)
}

const fn correct_deficit(a: i32, p: u32) -> i32 {
    a + (i32_sign_mask(a) & (p as i32))
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct ExtendedGCDResult {
    gcd: u32,
    bezout: [i32; 2],
}

const fn extended_gcd(a: u32, b: u32) -> ExtendedGCDResult {
    let mut old_r = a;
    let mut r = b;
    let mut old_s = 1;
    let mut s = 0;
    let mut old_t = 0;
    let mut t = 1;

    while r != 0 {
        let quotient = old_r / r;
        (old_r, r) = (r, old_r - quotient * r);
        (old_s, s) = (s, old_s - quotient as i32 * s);
        (old_t, t) = (t, old_t - quotient as i32 * t);
    }
    ExtendedGCDResult {
        gcd: old_r,
        bezout: [old_s, old_t],
    }
}

const fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b)
    }
    a
}

const SP_NBITS: u32 = u32::BITS - 2;
const PRE_SHIFT2: u32 = 2 * SP_NBITS + 1;
const POST_SHIFT: u32 = 1;

const fn used_bits(z: u32) -> u32 {
    u32::BITS - z.leading_zeros()
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct Z32Info {
    p: u32,
    p_inv: SpInverse64,
    red_struct: ReduceStruct,
}

impl Z32Info {
    const fn new(p: u32) -> Self {
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

const fn prep_mul_mod(p: u32) -> SpInverse64 {
    let shamt = p.leading_zeros() - (u32::BITS - SP_NBITS);
    let inv = normalised_prep_mul_mod(p << shamt);
    SpInverse64 { inv, shamt }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct ReduceStruct {
    ninv: u32,
    sgn: u32,
}

const fn prep_rem(p: u32) -> ReduceStruct {
    let mut q = (1 << (u32::BITS - 1)) / p;
    // r = 2^31 % p
    let r = (1 << (u32::BITS - 1)) - q * p;

    q *= 2;
    q += correct_excess_quo(2 * r as i32, p as i32).0;

    ReduceStruct { ninv: q, sgn: r }
}

const fn correct_excess_quo(a: i32, n: i32) -> (u32, i32) {
    if a >= n {
        (1, a - n)
    } else {
        (0, a)
    }
}

// TODO: make const as soon as allowed by rust
trait SignMask {
    fn sign_mask(self) -> i32;
}

impl SignMask for u32 {
    fn sign_mask(self) -> i32 {
        u32_sign_mask(self)
    }
}

impl SignMask for i32 {
    fn sign_mask(self) -> i32 {
        i32_sign_mask(self)
    }
}

const fn i32_sign_mask(i: i32) -> i32 {
    i >> (u32::BITS - 1)
}

const fn u32_sign_mask(i: u32) -> i32 {
    i32_sign_mask(i as i32)
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SpInverse64 {
    inv: u32,
    shamt: u32,
}

// Adapted from NTL's sp_NormalizedPrepMulMod
//
// Floating-point arithmetic replaced be u64 / i64 to allow `const`.
// The performance impact is not a huge concern since this function
// is only evaluated at compile time and only once for each prime field order.
// This is unlike NTL, where each change triggers a recalculation?
//
// This only works since this function is `const` and can be therefore
// used to compute individual `const INFO` inside `Z32<P>` for each
// `P`. The alternatives `lazy_static!` or `OnceCell` would not be
// recomputed, but instead incorrectly shared between `Z32<P>` with
// different `P`!
const fn normalised_prep_mul_mod(n: u32) -> u32 {
    // NOTE: this is an initial approximation
    //       the true quotient is <= 2^SP_NBITS
    const MAX: u64 = 1u64 << (2 * SP_NBITS - 1);
    let init_quot_approx = MAX / n as u64;

    let approx_rem = MAX - n as u64 * init_quot_approx;

    let approx_rem = (approx_rem << (PRE_SHIFT2 - 2 * SP_NBITS + 1)) - 1;

    let approx_rem_low = approx_rem as u32;
    let s1 = (approx_rem >> u32::BITS) as u32;
    let s2 = approx_rem_low >> (u32::BITS - 1);
    let approx_rem_high = s1.wrapping_add(s2);

    let approx_rem_low = approx_rem_low as i32;
    let approx_rem_high = approx_rem_high as i32;

    let bpl = 1i64 << u32::BITS;

    let fr = approx_rem_low as i64 + approx_rem_high as i64 * bpl;

    // now convert fr*ninv to a long
    // but we have to be careful: fr may be negative.
    // the result should still give floor(r/n) pm 1,
    // and is computed in a way that avoids branching

    let mut q1 = (fr / n as i64) as i32;
    if q1 < 0 {
        // This counteracts the round-to-zero behavior of conversion
        // to i32.  It should be compiled into branch-free code.
        q1 -= 1
    }

    let mut q1 = q1 as u32;
    let approx_rem_low = approx_rem_low as u32;
    let sub = q1.wrapping_mul(n);

    let approx_rem = approx_rem_low.wrapping_sub(sub);

    q1 += (1
        + u32_sign_mask(approx_rem)
        + u32_sign_mask(approx_rem.wrapping_sub(n))) as u32;

    ((init_quot_approx as u32) << (PRE_SHIFT2 - 2 * SP_NBITS + 1))
        .wrapping_add(q1)

    // NTL_PRE_SHIFT1 is 0, so no further shift required
}

#[cfg(test)]
mod tests {

    use ::rand::{Rng, SeedableRng};
    use once_cell::sync::Lazy;
    use rug::{ops::Pow, Integer};

    use super::*;

    const PRIMES: [u32; 3] = [3, 65521, 1073741789];

    #[test]
    fn z32_has_inv() {
        type Z = Z32<6>;
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
    fn z32_inv0() {
        type Z = Z32<6>;
        Z::from(0).inv();
    }

    #[test]
    #[should_panic]
    fn z32_inv2() {
        type Z = Z32<6>;
        Z::from(2).inv();
    }

    #[test]
    fn z32_constr() {
        let z: Z32<3> = 2.into();
        assert_eq!(u32::from(z), 2);
        let z: Z32<3> = (-1).into();
        assert_eq!(u32::from(z), 2);
        let z: Z32<3> = 5.into();
        assert_eq!(u32::from(z), 2);

        let z: Z32<3> = 0.into();
        assert_eq!(u32::from(z), 0);
        let z: Z32<3> = 3.into();
        assert_eq!(u32::from(z), 0);

        let z: Z32<3> = 2u32.into();
        assert_eq!(u32::from(z), 2);
        let z: Z32<3> = 5u32.into();
        assert_eq!(u32::from(z), 2);

        let z: Z32<3> = 0u32.into();
        assert_eq!(u32::from(z), 0);
        let z: Z32<3> = 3u32.into();
        assert_eq!(u32::from(z), 0);
    }

    static POINTS: Lazy<[i32; 1000]> = Lazy::new(|| {
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
            let z: Z32<{ PRIMES[0] }> = pt.into();
            let z: i32 = z.into();
            assert_eq!(z, pt.rem_euclid(PRIMES[0] as i32));
        }

        for pt in *POINTS {
            let z: Z32<{ PRIMES[1] }> = pt.into();
            let z: i32 = z.into();
            assert_eq!(z, pt.rem_euclid(PRIMES[1] as i32));
        }

        for pt in *POINTS {
            let z: Z32<{ PRIMES[2] }> = pt.into();
            let z: i32 = z.into();
            assert_eq!(z, pt.rem_euclid(PRIMES[2] as i32));
        }
    }

    #[test]
    fn tst_add() {
        for pt1 in *POINTS {
            let z1: Z32<{ PRIMES[0] }> = pt1.into();
            let pt1 = pt1 as i64;
            for pt2 in *POINTS {
                let z2: Z32<{ PRIMES[0] }> = pt2.into();
                let pt2 = pt2 as i64;
                let sum1: i32 = (z1 + z2).into();
                let sum2 = (pt1 + pt2).rem_euclid(PRIMES[0] as i64) as i32;
                assert_eq!(sum1, sum2);
            }
        }

        for pt1 in *POINTS {
            let z1: Z32<{ PRIMES[1] }> = pt1.into();
            let pt1 = pt1 as i64;
            for pt2 in *POINTS {
                let z2: Z32<{ PRIMES[1] }> = pt2.into();
                let pt2 = pt2 as i64;
                let sum1: i32 = (z1 + z2).into();
                let sum2 = (pt1 + pt2).rem_euclid(PRIMES[1] as i64) as i32;
                assert_eq!(sum1, sum2);
            }
        }

        for pt1 in *POINTS {
            let z1: Z32<{ PRIMES[2] }> = pt1.into();
            let pt1 = pt1 as i64;
            for pt2 in *POINTS {
                let z2: Z32<{ PRIMES[2] }> = pt2.into();
                let pt2 = pt2 as i64;
                let sum1: i32 = (z1 + z2).into();
                let sum2 = (pt1 + pt2).rem_euclid(PRIMES[2] as i64) as i32;
                assert_eq!(sum1, sum2);
            }
        }
    }

    #[test]
    fn tst_sub() {
        for pt1 in *POINTS {
            let z1: Z32<{ PRIMES[0] }> = pt1.into();
            let pt1 = pt1 as i64;
            for pt2 in *POINTS {
                let z2: Z32<{ PRIMES[0] }> = pt2.into();
                let pt2 = pt2 as i64;
                let sum1: i32 = (z1 - z2).into();
                let sum2 = (pt1 - pt2).rem_euclid(PRIMES[0] as i64) as i32;
                assert_eq!(sum1, sum2);
            }
        }

        for pt1 in *POINTS {
            let z1: Z32<{ PRIMES[1] }> = pt1.into();
            let pt1 = pt1 as i64;
            for pt2 in *POINTS {
                let z2: Z32<{ PRIMES[1] }> = pt2.into();
                let pt2 = pt2 as i64;
                let sum1: i32 = (z1 - z2).into();
                let sum2 = (pt1 - pt2).rem_euclid(PRIMES[1] as i64) as i32;
                assert_eq!(sum1, sum2);
            }
        }

        for pt1 in *POINTS {
            let z1: Z32<{ PRIMES[2] }> = pt1.into();
            let pt1 = pt1 as i64;
            for pt2 in *POINTS {
                let z2: Z32<{ PRIMES[2] }> = pt2.into();
                let pt2 = pt2 as i64;
                let sum1: i32 = (z1 - z2).into();
                let sum2 = (pt1 - pt2).rem_euclid(PRIMES[2] as i64) as i32;
                assert_eq!(sum1, sum2);
            }
        }
    }

    #[test]
    fn tst_mul() {
        for pt1 in *POINTS {
            let z1: Z32<{ PRIMES[0] }> = pt1.into();
            let pt1 = pt1 as i64;
            for pt2 in *POINTS {
                let z2: Z32<{ PRIMES[0] }> = pt2.into();
                let pt2 = pt2 as i64;
                let prod1: i32 = (z1 * z2).into();
                let prod2 = (pt1 * pt2).rem_euclid(PRIMES[0] as i64) as i32;
                assert_eq!(prod1, prod2);
            }
        }

        for pt1 in *POINTS {
            let z1: Z32<{ PRIMES[1] }> = pt1.into();
            let pt1 = pt1 as i64;
            for pt2 in *POINTS {
                let z2: Z32<{ PRIMES[1] }> = pt2.into();
                let pt2 = pt2 as i64;
                let prod1: i32 = (z1 * z2).into();
                let prod2 = (pt1 * pt2).rem_euclid(PRIMES[1] as i64) as i32;
                assert_eq!(prod1, prod2);
            }
        }

        for pt1 in *POINTS {
            let z1: Z32<{ PRIMES[2] }> = pt1.into();
            let pt1 = pt1 as i64;
            for pt2 in *POINTS {
                let z2: Z32<{ PRIMES[2] }> = pt2.into();
                let pt2 = pt2 as i64;
                let prod1: i32 = (z1 * z2).into();
                let prod2 = (pt1 * pt2).rem_euclid(PRIMES[2] as i64) as i32;
                assert_eq!(prod1, prod2);
            }
        }
    }

    #[test]
    fn tst_div() {
        for pt1 in *POINTS {
            let z1: Z32<{ PRIMES[0] }> = pt1.into();
            for pt2 in *POINTS {
                let z2: Z32<{ PRIMES[0] }> = pt2.into();
                if i32::from(z2) == 0 {
                    continue;
                }
                let div = z1 / z2;
                assert_eq!(z1, div * z2);
            }
        }

        for pt1 in *POINTS {
            let z1: Z32<{ PRIMES[1] }> = pt1.into();
            for pt2 in *POINTS {
                let z2: Z32<{ PRIMES[1] }> = pt2.into();
                if i32::from(z2) == 0 {
                    continue;
                }
                let div = z1 / z2;
                assert_eq!(z1, div * z2);
            }
        }

        for pt1 in *POINTS {
            let z1: Z32<{ PRIMES[2] }> = pt1.into();
            for pt2 in *POINTS {
                let z2: Z32<{ PRIMES[2] }> = pt2.into();
                if i32::from(z2) == 0 {
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
                let ref_pow0: u32 = ref_pow0.try_into().unwrap();
                let z: Z32<{ PRIMES[0] }> = pt1.into();
                let pow0: u32 = z.powu(exp as u64).into();
                assert_eq!(pow0, ref_pow0);

                let ref_pow0 =
                    (pow.clone() % PRIMES[1] + PRIMES[1]) % PRIMES[1];
                let ref_pow0: u32 = ref_pow0.try_into().unwrap();
                let z: Z32<{ PRIMES[1] }> = pt1.into();
                let pow0: u32 = z.powu(exp as u64).into();
                assert_eq!(pow0, ref_pow0);

                let ref_pow0 = (pow % PRIMES[2] + PRIMES[2]) % PRIMES[2];
                let ref_pow0: u32 = ref_pow0.try_into().unwrap();
                let z: Z32<{ PRIMES[2] }> = pt1.into();
                let pow0: u32 = z.powu(exp as u64).into();
                assert_eq!(pow0, ref_pow0);
            }
        }
    }
}
